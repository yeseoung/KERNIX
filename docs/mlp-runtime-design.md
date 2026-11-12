# KERNIX MLP 추론 런타임 설계

날짜: 2026-11-10

> 검증 절의 `test/` 파일과 테스트용 `make` 타깃은 개발 중에만 쓰고 공개 저장소에는 넣지 않았습니다. 그 절은 실행 기록입니다.

## 배경과 목표

KERNIX의 목표는 운영체제 수준에서 AI 연산을 보조하는 것이다. 지금은 2차원 float 텐서 연산 라이브러리이고, 모델을 불러와 실행하는 부분이 없다.

이 작업은 그 첫 단계로, PyTorch에서 학습한 MLP를 파일로 내보내 Mac에서 추론하는 런타임을 추가한다. Python이나 다른 프레임워크 없이 C API만으로 동작한다.

### 성공 기준

- 출력이 PyTorch와 최대 절대 오차 1e-4 이내로 일치한다.
- `make bench-model`에서 배치 1, 64, 4096의 forward 중앙값이 PyTorch 최선 백엔드(CPU와 MPS 중 빠른 쪽)보다 빠르다. 느린 구간이 있으면 숨기지 않고 결과에 그대로 적는다.
- forward 실행 중 메모리 할당이 없다(같은 배치 크기로 두 번째 호출부터).

### 범위 밖

학습, 합성곱, 어텐션, N차원 텐서, ONNX/GGUF, 양자화, 시스템 데몬/서비스화, `info.md`의 기존 버그 10건.

## 왜 런타임이 필요한가

기존 연산을 이어 붙이면(`float_matmul_metal → floats_add → floats_relu`) Linear 레이어 하나에 할당 3번, 메모리 왕복 3번이 든다. 런타임은 세 가지로 이를 없앤다.

1. **전치 제거**: 가중치를 `입력×출력` 방향으로 저장한다. 전치는 내보내기 스크립트가 한 번 한다.
2. **버퍼 재사용**: 레이어별 출력 버퍼를 배치 크기에 맞춰 한 번 할당하고 계속 쓴다. 가중치와 출력 버퍼는 `kernix_float_new`로 할당하므로 64KB 이상이면 Metal 공유 버퍼가 되어 GPU가 복사 없이 쓴다.
3. **융합**: `Linear + bias + 활성화`를 matmul 결과 버퍼 위에서 제자리로 처리한다. 32KB 구간 단위라 값이 캐시에 있는 동안 끝난다.

## 구성 요소

### 1. `include/kernix_model.h` (새 파일)

기존 방식대로 `static inline` 헤더다. `KERNIX_USE_ACCELERATE`가 필요하고, `KERNIX_USE_METAL`은 선택이다.

```c
typedef struct kernix_model kernix_model;

kernix_model* kernix_model_load(const char* path);             /* 실패 시 NULL + stderr */
const float_tensor* kernix_model_forward(kernix_model* m, const float_tensor* x); /* 실패 시 NULL */
void kernix_model_free(kernix_model* m);
```

- `forward`의 입력은 `배치×입력크기`, 출력은 `배치×출력크기`다.
- 반환된 텐서는 모델 소유다. 호출자가 해제하지 않고, 다음 `forward`가 덮어쓴다.
- 한 모델의 `forward`를 여러 스레드가 동시에 부르면 안 된다. 서로 다른 모델은 동시에 써도 된다.
- 입력 텐서는 수정하지 않는다.

내부 단계는 로드 시 정해진다.

| 단계 | 동작 | 버퍼 |
|---|---|---|
| LINEAR (+융합된 활성화) | matmul을 자기 출력 버퍼에 쓰고, 같은 버퍼에서 bias와 활성화 적용 | 자기 출력 버퍼 |
| ACT (Linear 뒤가 아닌 경우) | 현재 버퍼에서 제자리 | 없음 |
| LAYERNORM | 현재 버퍼에서 제자리 | 없음 |
| SOFTMAX | 현재 버퍼에서 제자리 | 없음 |

첫 단계가 제자리 단계면 입력을 수정하게 되므로, 그 경우에만 입력을 복사할 버퍼를 하나 둔다.

배치 크기가 바뀌면 버퍼를 다시 할당한다.

### 2. 제자리 커널

- `kernix_bias_act(buf, rows, cols, bias, act)`: `kernix_par`로 행을 나누고, 행마다 `vDSP_vadd`(bias) 뒤 활성화를 적용한다. `bias`는 NULL일 수 있고 `act`는 "없음"일 수 있다.
- 활성화 본문은 기존 `KERNIX_MAP` 본문과 같은 수식이다. ReLU, Sigmoid, Tanh, GELU는 입력과 출력이 같아도 된다. SiLU는 원본 값이 필요해서 `KERNIX_CHUNK` 크기 스택 임시 버퍼를 쓴다(기존 GELU와 같은 방식). 열 수가 `KERNIX_CHUNK`보다 크면 행을 그 크기 조각으로 나눠 처리한다.
- GELU는 torch 기본(정확한 erf) 방식만 지원한다.
- softmax와 layer_norm은 `KERNIX_ROWS` 매크로를 고쳐, 주어진 버퍼에 쓰는 `이름_into(in, out, rows, cols, ...)`를 함께 만든다. 기존 할당 함수는 이 `_into`를 부른다. 두 본문 모두 입력과 출력이 같아도 된다.

큰 텐서에서 bias+활성화를 GPU로 보내는 경로는 만들지 않는다. MLP의 `배치×은닉 크기`는 대개 GPU 하한(4M 원소)보다 작다.

### 3. `kernix_matmul_into` (기존 코드 변경)

`src/kernix_metal.m`의 `float_matmul_metal` 본문을 `int kernix_matmul_into(a, b, c)`로 옮긴다. `c`는 호출자가 준 `a->row × b->col` 텐서다. `float_matmul_metal`은 할당 후 이 함수를 부르는 래퍼가 된다. 기존 호출부의 동작은 그대로다.

Metal이 없는 빌드에서는 헤더의 같은 이름 함수가 `cblas_sgemm`을 직접 부른다.

### 4. 파일 형식 `.knx`

리틀 엔디언, 값은 모두 4바이트다.

```
"KNX1"                       매직
uint32 n_ops                 1 이상 1024 이하
op마다:
  uint32 kind                0 LINEAR, 1 ACT, 2 LAYERNORM, 3 SOFTMAX
  LINEAR:    uint32 in, out, has_bias
             float32 W[in*out]   (in×out, 행 우선)
             float32 b[out]      (has_bias일 때)
  ACT:       uint32 act      0 relu, 1 gelu, 2 silu, 3 sigmoid, 4 tanh
  LAYERNORM: uint32 cols, has_affine; float32 eps
             float32 gamma[cols], beta[cols]  (has_affine일 때)
  SOFTMAX:   (없음)
```

### 5. `test/export_mlp.py` (지금은 `tools/export_torch.py`)

`torch.nn.Sequential`을 `.knx`로 쓴다. 지원하지 않는 레이어, `approximate='tanh'` GELU, 마지막 차원이 아닌 Softmax는 예외로 거부한다. Linear 가중치는 `W.T`를 연속 메모리로 써서 낸다.

## 오류 처리

로드는 신뢰할 수 없는 파일을 읽는 경계다. 다음은 모두 `NULL`을 반환하고 stderr에 이유를 적는다.

- 파일 열기 실패, 매직 불일치, `n_ops` 범위 밖
- 알 수 없는 kind/act, 차원 0, `in*out` 곱의 오버플로 또는 `int` 범위 초과
- 파일이 선언된 크기보다 짧음(`fread` 개수 확인)
- 차원 연결 불일치: LINEAR의 `in` 또는 LAYERNORM의 `cols`가 앞 단계 출력 크기와 다름
- 할당 실패

`forward`는 입력이 NULL이거나 열 수가 모델 입력 크기와 다르면 `NULL`을 반환한다. 배치 0은 `0×출력크기` 텐서를 반환한다.

## 검증

`make bench-model`이 다음을 차례로 실행한다.

1. `test/mlp_bench.py`(처음에는 `export_mlp.py`)가 고정 시드로 모델 두 개를 만든다. 하나는 모든 레이어 종류를 쓰는 작은 모델(정확성용), 하나는 `784 → 1024 → 1024 → 10` ReLU MLP(속도용)다. 모델, 입력, 기대 출력을 `test/model_data/`에 쓴다(gitignore).
2. `test/model_test.c --check`가 다음을 검사한다.
   - 배치 1, 37, 4096에서 기대 출력과 최대 절대 오차 1e-4 이내
   - 같은 모델로 배치 크기를 바꿔 가며 반복 호출해도 결과가 같음(버퍼 재사용)
   - 입력 텐서가 바뀌지 않음
   - 잘린 파일, 틀린 매직, 차원이 안 맞는 파일은 로드가 NULL
   - 열 수가 틀린 입력은 forward가 NULL
3. 같은 실행 파일이 배치 1 / 64 / 4096의 forward 중앙값(20회)을 출력하고, Python 스크립트가 같은 모델의 PyTorch CPU·MPS 시간과 나란히 보여 준다.

## 알려진 한계

- 배치 크기별 버퍼는 한 벌만 유지한다. 배치 크기를 번갈아 쓰면 매번 재할당한다.
- 행렬 곱의 GPU 하한과 분할 비율은 M1 실측값이다(기존 코드와 같음).
