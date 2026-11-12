# KERNIX 그래프 런타임 설계 (KNX3)

> 상태: 구현됨 · 작성 2026-11-12 · 대상 브랜치 `feat/model-runtime`
> 이전 단계: [MLP 런타임 (KNX1)](mlp-runtime-design.md), [int8 양자화 (KNX2)](int8-quantization-design.md)
> 검증 절의 `test/` 파일과 테스트용 `make` 타깃은 개발 중에만 쓰고 공개 저장소에는 넣지 않았습니다. 그 절은 실행 기록입니다.

## 1. 목적

KNX1·KNX2는 연산이 일렬로 이어지는 MLP와 LSTM만 표현합니다. KNX3는 다음 모델을 Keras와 PyTorch에서 내보내 실행합니다.

- **CNN**: Conv1D/2D(보폭, 패딩, dilation, groups, depthwise, separable), 풀링, BatchNorm
- **RNN**: LSTM, GRU, SimpleRNN, 양방향, 여러 층
- **Transformer**: 멀티헤드 어텐션(셀프·교차·인과 마스크), 잔차 연결, 임베딩

결정 사항:
- N차원 텐서는 **런타임 안에서만** 다룹니다. `kernix.h`의 2차원 `float_tensor`와 연산 27종은 그대로 둡니다.
- 기존 C API, Python 바인딩, KNX1/KNX2 파일은 그대로 동작합니다.

## 2. 값의 표현

그래프의 값(텐서) 하나는 표본 하나의 **channels-last 모양**을 갖습니다. 축은 최대 4개이고, 마지막 축이 채널입니다.

| 모델 | 표본 모양 | 버퍼 (2차원) |
|---|---|---|
| MLP | `(F)` | 배치 × F |
| 시퀀스 | `(T, F)` | (배치·T) × F |
| 이미지 | `(H, W, C)` | (배치·H·W) × C |

버퍼는 항상 `(배치 × 앞 축들의 곱) × 채널`의 2차원입니다. 그래서 다음이 성립합니다.

- **Linear**는 마지막 축에 대한 행렬곱 하나입니다. Keras `Dense`를 3차원 입력에 쓰는 경우도 같습니다.
- **합성곱**은 im2col 뒤 행렬곱입니다. 출력 행을 2 MB 조각으로 나눠 4갈래가 병렬로 조각마다 im2col → sgemm(AMX) → bias·act를 합니다. 펼친 행렬이 캐시에 남아 메모리를 왕복하지 않습니다.
  - im2col 버퍼는 모델 하나에 하나(`m->col`)이고 모든 합성곱이 공유합니다. 크기는 조각 단위라 배치와 무관하고, 작업 버퍼 상한에서는 고정분(`ws_fixed`)으로 셉니다.
  - 1×1 보폭 1은 im2col 없이 입력을 그대로 씁니다.
  - depthwise는 채널 방향 벡터 곱-합으로 따로 처리합니다.
- **Flatten과 Reshape**은 복사 없이 보기만 바꿉니다.
- **어텐션**은 배치·T행을 그대로 query, key, value 투영에 넣습니다.

PyTorch의 channels-first(NCHW)는 내보낼 때 맞춥니다(§5.2).

## 3. 파일 형식 `.knx` v3

리틀 엔디언입니다. 가중치 블록은 KNX2와 같습니다(`uint32 dtype` + F32/I8_COL/I8_ROW).

```
"KNX3"
uint32 n_ops                    1 이상 1024 이하
uint32 layout                   0 = channels-last, 1 = PyTorch channels-first 입력 (바인딩이 축을 옮김)
uint32 in_nd, in_dims[in_nd]    표본 하나의 입력 모양 (channels-last)
연산마다:
  uint32 kind
  uint32 n_in, in[n_in]         입력 값 번호: 0 = 모델 입력, k+1 = 연산 k의 출력 (앞 연산만 가리킴)
  인자와 가중치 (kind별)
```

모델 출력은 마지막 연산의 값입니다.

| kind | 연산 | 입력 수 | 인자 · 가중치 |
|---:|---|---|---|
| 0 | LINEAR | 1 | in, out, has_bias · W(in×out), b |
| 1 | ACT | 1 | act, f32 alpha |
| 2 | LAYERNORM | 1 | cols, affine, f32 eps · γ, β |
| 3 | SOFTMAX | 1 | (마지막 축) |
| 4/5/6 | LSTM / GRU / RNN | 1 | in, units, act, ret_seq, dirs, merge · 방향마다 W(in×G·u), U(u×G·u), bias |
| 7 | CONV | 1 | cin, cout, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, groups, has_bias · W((kh·kw·cin/g)×cout), b |
| 8 | POOL | 1 | type(max/avg), kh, kw, sh, sw, pt, pb, pl, pr, count_pad |
| 9 | GLOBAL_POOL | 1 | type |
| 10 | RESHAPE | 1 | nd, dims (보기) |
| 11 | PERMUTE | 1 | nd, perm (출력 축 k = 입력 축 perm[k]) |
| 12 | BINARY | 2 | op (add, sub, mul, div, max, min), 행 또는 채널 방향 크기 1은 브로드캐스트 |
| 13 | CONCAT | 2~8 | axis (첫 축 / 마지막 축) |
| 14 | EMBEDDING | 1 | vocab, dim · 표 (id가 범위 밖이거나 정수가 아니면 NaN 행) |
| 15 | AFFINE | 1 | c · scale, shift (채널마다: BatchNorm, Normalization, Rescaling, 상수 곱·덧셈) |
| 16 | ADD_CONST | 1 | n · 표본 하나 크기의 상수 (위치 임베딩) |
| 17 | ATTENTION | 1~3 | heads, dk, dv, causal, cq, cv, ck, cout, has_bias · Wq, bq, Wk, bk, Wv, bv, Wo, bo |
| 18 | SELECT | 1 | axis, start, len, squeeze (`x[:, -1]`, Cropping) |
| 19 | PAD | 1 | pt, pb, pl, pr (0으로) |
| 20 | UPSAMPLE | 1 | sh, sw (nearest) |

연산 세부 규칙:
- **RNN 게이트 순서**는 Keras를 따릅니다. LSTM은 i, f, c, o, GRU는 z, r, h(reset_after)입니다. GRU의 bias는 입력 쪽 3u 다음에 순환 쪽 3u가 옵니다. PyTorch GRU(r, z, n)는 내보낼 때 순서를 바꿉니다.
- **역방향**은 t = T−1부터 읽고, 출력은 같은 시점 위치에 둡니다. Keras Bidirectional, PyTorch와 같습니다.
- **ATTENTION**의 입력은 (query[, value[, key]])입니다. key가 없으면 value를, value가 없으면 query를 씁니다.
  - 헤드마다 `softmax(Q·Kᵀ/√dk [+ 인과 마스크])·V`를 계산하고, 결과를 이어 붙여 `·Wo + bo`를 합니다.
- **활성화** 16종: relu, gelu, silu, sigmoid, tanh, leaky_relu(α), elu(α), selu, softplus, relu6, hard_sigmoid, hard_swish, gelu_tanh, exp, mish, softsign.
  - CPU 경로(`kernix_act_chunk`)와 GPU 후처리 커널(`bias_act`)이 같은 수식을 씁니다.

## 4. 로더와 실행 (`include/kernix_model.h`)

### 4.1 로드 단계

1. **파싱**: 인자와 가중치를 읽습니다. 파일은 신뢰할 수 없는 입력이라 다음을 거부합니다.
   - 남은 바이트보다 큰 선언
   - 크기 상한 초과
   - 잘못된 입력 번호, 알 수 없는 kind/act/dtype
2. **모양 추론**(`kernix_infer`): 값마다 모양을 계산합니다. 다음은 거부합니다.
   - 차원 불일치, 1차원 입력에 2차원 커널, 빈 출력
   - 브로드캐스트 불가, int 범위 초과
3. **융합**: LINEAR/CONV의 출력을 ACT 하나만 읽으면 활성화를 합칩니다.
4. **버퍼 계획**(`kernix_plan`): 값마다 저장 버퍼를 정합니다.
   - 수명이 끝난 버퍼는 다음 값이 다시 씁니다(크기가 맞는 것 중 가장 작은 것).
   - RESHAPE은 입력 버퍼를 그대로 봅니다.
   - ACT, LAYERNORM, SOFTMAX, AFFINE, ADD_CONST, BINARY는 입력을 마지막으로 읽는 경우 입력 버퍼에 제자리로 씁니다.
   - 호출자 입력은 읽기만 합니다.

### 4.2 실행

- 조각마다 값별 보기(`views`)를 다시 잡고 연산을 차례로 실행합니다.
- 배치 크기가 같으면 forward 중 할당이 없습니다.
- 작업 버퍼 상한(나눠 실행)과 `trim`은 KNX1/KNX2와 같습니다.
- `ws_per_sample`에는 저장 버퍼와 연산별 작업 버퍼(RNN 게이트, 어텐션 점수)가 들어갑니다. 공유 im2col 버퍼는 `ws_fixed`에 들어갑니다.

### 4.3 호환

- **KNX1/KNX2**: 같은 내부 구조로 읽습니다. 연산마다 앞 값을 입력으로 삼고, 입력 모양은 (첫 LSTM의 T, 입력 크기) 또는 (입력 크기)입니다.
- **기존 필드 유지**: `m->steps[i].w/q/rec/deq`, `n_steps`, `ws_per_sample` 등은 이름과 의미를 그대로 둬서 기존 검사가 그대로 돕니다.
- **C API**: `kernix_capi_shapes`(표본의 입력·출력 모양, layout)를 추가했습니다.
- **Python `Model`**:
  - `(batch, *in_shape)`를 받고 `(batch, *out_shape)`를 돌려줍니다.
  - 이전 방식인 `(rows, in_dim)`도 받습니다.
  - PyTorch 합성곱 모델(layout 1)은 `(batch, C, H, W)`로 주고받습니다.

## 5. 내보내기

| 프레임워크 | 명령 | 방식 |
|---|---|---|
| Keras | `python tools/convert.py 모델.h5 출력.knx [--quantize int8]` | 일렬 LSTM/Dense는 KNX1/KNX2(기존과 같은 바이트), 그 밖은 `tools/keras_graph.py`가 함수형 그래프를 따라 KNX3 |
| PyTorch | `python tools/convert.py 모델.pt 출력.knx [--quantize int8] [--input-shape 3,32,32]` | 일렬 MLP는 KNX1, 그 밖은 `tools/torch_graph.py`가 `torch.fx`로 KNX3. 첫 층이 Linear가 아니면 `--input-shape` 필요 |

두 프레임워크 모두 `tools/knx3.py`의 `Graph`로 씁니다. `Graph`는 C와 같은 규칙으로 모양을 추적합니다. 쓰기 직전에는 LINEAR/CONV 바로 뒤의 AFFINE(BatchNorm 등)을 그 출력을 AFFINE만 읽는 경우에 한해 가중치에 접어 넣습니다.

### 5.1 Keras

**지원하는 레이어**
- **Dense 계열**: Dense, TimeDistributed(Dense)
- **활성화**: Activation(16종), ReLU(max_value 6, negative_slope), LeakyReLU, ELU, PReLU(채널마다), Softmax
- **정규화**: BatchNormalization, LayerNormalization, Normalization, Rescaling
- **합성곱**: Conv1D/2D(valid/same/causal, groups), DepthwiseConv1D/2D, SeparableConv1D/2D
- **풀링**: Max/AveragePooling1D/2D, Global*Pooling1D/2D(keepdims)
- **모양**: Flatten, Reshape, Permute, ZeroPadding, UpSampling(nearest), Cropping
- **합치기**: Add, Subtract, Multiply, Maximum, Minimum, Average, Concatenate(첫 축/마지막 축)
- **시퀀스**: Embedding, LSTM, GRU(reset_after), SimpleRNN, Bidirectional(concat/sum/mul/ave), MultiHeadAttention(인과 마스크, 교차, value_dim, output_shape)
- **그 밖**: Dropout 계열은 건너뜁니다. 중첩 모델도 지원합니다.

**거부하는 것**
- channels_first 데이터
- attention_mask 텐서, Embedding mask_zero
- 상태 유지 RNN, go_backwards 단독, GRU reset_after=False
- Lambda, 입력 모양이 정해지지 않은 모델

### 5.2 PyTorch

**축 배치 맞추기**

값마다 "논리 축 i(PyTorch, 배치 제외)가 저장 축 lperm[i]에 있다"를 들고 다닙니다.
- `permute`와 `transpose`는 이 대응만 바꾸고, 실제 연산은 넣지 않습니다.
- 연산이 특정 배치를 요구할 때만 PERMUTE를 넣습니다.
  - Conv/Pool/BatchNorm은 채널이 저장 마지막 축이어야 합니다.
  - Linear/LayerNorm/softmax는 그 축이 마지막이어야 합니다.
  - flatten/view는 PyTorch 순서여야 합니다.
- 첫 소비자가 합성곱이면 입력 layout을 1(channels-first)로 정해서, 바인딩이 `(B, C, H, W)`를 받게 합니다.

**그 밖의 규칙**
- 예제 입력으로 한 번 실행해 노드마다 실제 값을 기록합니다. 입력과 무관한 노드(파라미터, 버퍼, `arange`, `size()`)는 상수가 됩니다.
- 상수와의 연산은 다음처럼 바꿉니다.
  - 스칼라·채널마다의 곱이나 덧셈 → AFFINE
  - 표본 크기 상수의 덧셈(위치 파라미터) → ADD_CONST
- 출력에 쓰이지 않는 노드는 처리하지 않습니다. 예: `_, (h, _) = lstm(x)`의 `c_n`

**지원하는 것**
- **모듈**
  - 기본: Linear, Conv1d/2d(groups, `padding='same'`), BatchNorm1d/2d, LayerNorm, Embedding
  - 풀링: Max/AvgPool1d/2d, AdaptiveAvg/MaxPool(출력 1 또는 나누어떨어지는 크기)
  - 모양: Flatten/Unflatten, Upsample(nearest), ZeroPad/ConstantPad(0)
  - 활성화 14종, PReLU, Softmax, Dropout, Identity
  - 시퀀스:
    - LSTM/GRU/RNN(batch_first, 여러 층, 양방향). `out[:, t]`와 `h_n[-1]`(양방향은 `h_n[-2]`도)을 쓸 수 있습니다.
    - MultiheadAttention(batch_first, kdim/vdim, 인과 마스크)
    - TransformerEncoderLayer/Encoder(norm_first, relu/gelu, 인과 마스크)
- **함수와 메서드**
  - 산술: `+ − × ÷`, `torch.cat`
  - 모양: flatten/view/reshape/squeeze/unsqueeze, permute/transpose, `x[:, i]`와 `x[:, a:b]`(첫 축·마지막 축)
  - 축 줄이기: mean/amax(축 하나만 남기는 경우)
  - 그 밖: F.pad, F.interpolate(nearest), F의 활성화와 softmax

**거부하는 것**
- key_padding_mask, 인과가 아닌 어텐션 마스크
- 입력에 따라 바뀌는 마스크
- GroupNorm, ConvTranspose
- 가운데 축 인덱싱, 배치 축을 옮기는 permute

## 6. 검증

| 명령 | 내용 |
|---|---|
| `make bench-model` | 기존 KNX1/KNX2 검사 전부 (Metal·Accelerate 빌드): 로더 거부 사례, forward, LSTM, 나눠 실행·trim, int8 |
| `make test-zoo` | PyTorch 모델 17개 (위 목록의 모든 종류 + int8): KERNIX 출력과 PyTorch 출력의 상대 오차 ≤ 2e-4 |
| `make test-zoo-keras SIGN_PROJECT=…` | Keras 모델 21개 (위 목록의 모든 종류 + int8): 같은 기준 |
| `make test-fuzz` | 정상 KNX3 모델 3개(CNN, RNN, 어텐션)를 자르거나 바꾼 파일 4,500개를 ASan/UBSan 빌드에 넣음 |
| `make bench-matmul-metal` | GPU 후처리 커널: 활성화 6종 × bias 유무를 CPU 결과와 비교 |

2026-11-12 결과:
- 모델 모음은 모두 통과했습니다. 상대 오차는 int8을 빼면 1e-7 수준입니다.
- 퍼징에서는 메모리 오류가 없었습니다. 2,665개는 거부됐고, 1,835개는 값만 바뀌어 로드와 실행까지 갔습니다.
  - 퍼징이 찾은 Permute 검사의 시프트 오류 하나를 고쳤습니다.

## 7. 성능 (`make bench-zoo`, M1, Python 바인딩, Accelerate 빌드)

| 모델 | 배치 | PyTorch CPU | PyTorch MPS | KERNIX | PyTorch 최선 대비 |
|---|---:|---:|---:|---:|---:|
| ResNet (32채널, 블록 4개 + SE), 3×32×32 | 1 | 1.21 ms | 3.40 ms | **0.46 ms** | **2.6배** |
| | 64 | 65.8 ms | 20.8 ms | **20.8 ms** | 1.0배 |
| GPT (임베딩 16, 헤드 4, 2층, 인과), 토큰 10개 | 1 | 0.27 ms | 1.21 ms | **0.019 ms** | **13.9배** |
| | 64 | 1.72 ms | 1.52 ms | **0.45 ms** | **3.3배** |

- 표본 1개 추론(실시간)은 연산당 오버헤드가 없어 크게 빠릅니다.
- 큰 배치 CNN은 조각 단위 im2col 이후(2026-11-12) CPU와 AMX만으로 MPS와 비슷합니다(39 → 21 ms, 최대 메모리 70 → 40 MB). Python 바인딩은 Accelerate 빌드라 GPU를 쓰지 않습니다.

## 8. 한계와 후속 작업

- im2col 버퍼는 합성곱끼리 공유하지만, 그 밖의 연산별 작업 버퍼(tmp: RNN 게이트, 어텐션 점수)는 연산끼리 공유하지 않습니다.
  - 공유하면 깊은 RNN·어텐션 모델의 메모리가 줄어듭니다. 필요해지면 연산별 크기의 최댓값으로 하나만 잡습니다.
- 다음은 지원하지 않습니다.
  - ConvTranspose, GroupNorm, 패딩 마스크
  - 동적 시퀀스 길이: 입력 모양은 파일에 고정되므로, 길이마다 내보내야 합니다.
- PyTorch 내보내기는 예제 입력 하나로 경로를 기록합니다. 입력 값에 따라 분기하는 모델은 그 경로만 담깁니다.
- 수제 GEMM(로드맵)은 이 작업에 넣지 않았습니다.
  - NEON은 이론 최대(성능 코어 4개 기준 약 0.4 TFLOPS)가 AMX 실측(약 0.9 TFLOPS)보다 낮습니다.
  - Metal은 MPS(실측 1.8 TFLOPS, 이론의 약 70%)를 넘어야 의미가 있어 별도 연구로 둡니다.
