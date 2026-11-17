"""Keras 모델 모음: 레이어 종류마다 작은 모델을 무작위 가중치로 만들어 .knx로 내보내고, KERNIX 출력이 Keras와 같은지 본다.
tensorflow가 있는 파이썬으로 실행 (make test-zoo-keras KERAS_PY=<그 파이썬>). 먼저 make python-lib
"""
import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

import keras
from keras import layers as L

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from export_keras import export  # noqa: E402
from kernix import Model  # noqa: E402

keras.utils.set_random_seed(0)
rng = np.random.default_rng(0)
fail = 0


def randomize(model):
    """BatchNorm 통계 등 기본값(0, 1)이 아닌 값으로: 접어 넣기·채널 순서 오류가 드러나게"""
    for w in model.weights:
        v = keras.ops.convert_to_numpy(w)
        if "moving_variance" in w.path or "variance" in w.path.split("/")[-1]:
            w.assign(rng.uniform(0.5, 2.0, v.shape))
        elif v.dtype.kind == "f" and "count" not in w.path:
            w.assign(rng.normal(0, 0.4, v.shape) if "embeddings" not in w.path else rng.normal(0, 1, v.shape))


def check(name, model, x, quantize=None, tol=2e-4):
    global fail
    randomize(model)
    want = np.asarray(model(x, training=False))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.knx")
        try:
            export(model, path, quantize)
            m = Model(path)
            got = m(x)
            fmt = open(path, "rb").read(4).decode()
        except Exception as e:  # noqa: BLE001  내보내기·실행 실패도 결과로 보고
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            fail += 1
            return
    err = np.abs(got.reshape(want.shape) - want).max() / max(1.0, np.abs(want).max())
    ok = got.shape == want.shape and err <= tol
    print(f"  {'✓' if ok else '✗'} {name} [{fmt}{' int8' if quantize else ''}]: 출력 {want.shape[1:]}, 상대 오차 {err:.1e}"
          + ("" if got.shape == want.shape else f" (모양 {got.shape} != {want.shape})"))
    fail += not ok


def seq(*ls, shape):
    return keras.Sequential([keras.Input(shape)] + list(ls))


def x_of(shape, n=5):
    return rng.normal(0, 1, (n,) + shape).astype(np.float32)


print("=== Keras 모델 모음 → KERNIX ===")
# MLP와 정규화
check("MLP (Dense, BatchNorm 접기, Dropout, softmax)", seq(L.Dense(32), L.BatchNormalization(), L.Activation("relu"), L.Dropout(0.5),
      L.Dense(16, activation="gelu"), L.Dense(5, activation="softmax"), shape=(12,)), x_of((12,)))
check("MLP 기존 형식 (Dense만)", seq(L.Dense(8, activation="relu"), L.Dense(3), shape=(6,)), x_of((6,)))
check("Normalization + Rescaling + LayerNorm", seq(L.Rescaling(0.5, 1.0), L.LayerNormalization(), L.Dense(7), shape=(9,)), x_of((9,)))
acts = ["elu", "selu", "softplus", "relu6", "hard_sigmoid", "hard_silu", "mish", "softsign", "leaky_relu", "exponential",
        "tanh", "swish", "sigmoid", "gelu"]
check("활성화 14종", seq(*[L.Dense(6, activation=a) for a in acts], shape=(6,)), x_of((6,)) * 0.5)
check("활성화 레이어 (LeakyReLU, ELU, ReLU6, PReLU)", seq(L.Dense(8), L.LeakyReLU(0.1), L.Dense(8), L.ELU(0.7), L.Dense(8),
      L.ReLU(max_value=6.0), L.Dense(8), L.PReLU(), shape=(5,)), x_of((5,)))

# CNN
check("CNN 2D (same/valid, 보폭, BN, Max/AvgPool, GAP)", seq(
    L.Conv2D(8, 3, padding="same"), L.BatchNormalization(), L.ReLU(), L.MaxPooling2D(2),
    L.Conv2D(12, 3, strides=2, padding="valid", activation="relu"), L.AveragePooling2D(2, padding="same"),
    L.GlobalAveragePooling2D(), L.Dense(4, activation="softmax"), shape=(17, 15, 3)), x_of((17, 15, 3)))
check("CNN 2D 비정사각 커널·dilation·groups", seq(L.Conv2D(8, (3, 1), dilation_rate=(2, 1), padding="same"),
      L.Conv2D(8, 3, groups=2, padding="same"), L.Flatten(), L.Dense(3), shape=(9, 8, 4)), x_of((9, 8, 4)))
check("Depthwise · Separable · Pad · UpSampling · Cropping", seq(
    L.ZeroPadding2D(((1, 0), (2, 1))), L.DepthwiseConv2D(3, depth_multiplier=2, activation="relu6"),
    L.SeparableConv2D(6, 3, padding="same"), L.UpSampling2D((2, 1)), L.Cropping2D(((1, 2), (0, 1))),
    L.GlobalMaxPooling2D(keepdims=True), L.Flatten(), shape=(8, 7, 3)), x_of((8, 7, 3)))
check("CNN 1D (causal, dilation, 보폭, GlobalMaxPool)", seq(L.Conv1D(6, 3, padding="causal", dilation_rate=2, activation="relu"),
      L.Conv1D(5, 4, strides=2), L.MaxPooling1D(2), L.UpSampling1D(2), L.ZeroPadding1D(1), L.Cropping1D((0, 1)),
      L.GlobalMaxPooling1D(), shape=(20, 4)), x_of((20, 4)))
check("DepthwiseConv1D + AveragePooling1D same", seq(L.DepthwiseConv1D(3, padding="same"), L.AveragePooling1D(3, 2, padding="same"),
      L.GlobalAveragePooling1D(), shape=(11, 5)), x_of((11, 5)))

# 그래프: 잔차, 합치기, SE 블록
inp = keras.Input((10, 10, 6))
h = L.Conv2D(6, 3, padding="same", activation="relu")(inp)
h = L.Conv2D(6, 3, padding="same")(h)
h = L.ReLU()(L.Add()([h, inp]))
se = L.GlobalAveragePooling2D()(h)
se = L.Dense(3, activation="relu")(se)
se = L.Reshape((1, 1, 6))(L.Dense(6, activation="sigmoid")(se))
h = L.Multiply()([h, se])
b2 = L.Conv2D(4, 1)(h)
h = L.Concatenate()([h, b2, L.Subtract()([h, h])])
check("ResNet 블록 + SE + Concatenate", keras.Model(inp, L.Dense(3)(L.Flatten()(h))), x_of((10, 10, 6)))

inp = keras.Input((6, 4))
a, b = L.Dense(4)(inp), L.Dense(4)(inp)
h = L.Concatenate(axis=1)([L.Maximum()([a, b]), L.Average()([a, inp]), L.Minimum()([b, inp])])
h = L.Permute((2, 1))(h)
check("Concatenate(axis=1) · Maximum · Average · Permute", keras.Model(inp, L.Reshape((-1,))(h)), x_of((6, 4)))

# RNN
check("GRU → Bidirectional LSTM → SimpleRNN", seq(L.GRU(8, return_sequences=True), L.Bidirectional(L.LSTM(6, return_sequences=True)),
      L.SimpleRNN(5), L.Dense(3), shape=(7, 4)), x_of((7, 4)))
for mode in ("sum", "mul", "ave"):
    check(f"Bidirectional GRU merge={mode}", seq(L.Bidirectional(L.GRU(5), merge_mode=mode), shape=(6, 3)), x_of((6, 3)))
check("LSTM 시퀀스 출력 + TimeDistributed(Dense)", seq(L.LSTM(6, return_sequences=True), L.TimeDistributed(L.Dense(2)), shape=(5, 3)),
      x_of((5, 3)))
check("Embedding → LSTM", seq(L.Embedding(20, 6), L.LSTM(5), L.Dense(4), shape=(9,)), rng.integers(0, 20, (5, 9)).astype(np.float32))
check("LSTM 기존 형식 (수어 모델 구성)", seq(L.LSTM(8, return_sequences=True), L.LSTM(4), L.Dense(3, activation="softmax"), shape=(6, 5)),
      x_of((6, 5)))

# Transformer
inp = keras.Input((8, 12))
h = inp
for causal in (False, True):
    a = L.MultiHeadAttention(3, 4)(h, h, use_causal_mask=causal)
    h = L.LayerNormalization()(L.Add()([h, a]))
    f = L.Dense(12)(L.Dense(24, activation="gelu")(h))
    h = L.LayerNormalization()(L.Add()([h, f]))
kv = L.Dense(10)(inp)
c = L.MultiHeadAttention(2, 5, value_dim=3, output_shape=7)(h, kv, kv)
check("Transformer 인코더 2층 (+인과 마스크) + 교차 어텐션", keras.Model(inp, L.Dense(4)(L.GlobalAveragePooling1D()(c))), x_of((8, 12)))

# int8
check("CNN int8", seq(L.Conv2D(8, 3, padding="same", activation="relu"), L.GlobalAveragePooling2D(), L.Dense(4), shape=(8, 8, 3)),
      x_of((8, 8, 3)), quantize="int8", tol=5e-2)

# Masking은 패딩 시점을 건너뛰게 하는데 KERNIX에는 마스크가 없음: 항등으로 넘기면 패딩 입력에서 조용히 틀리므로 거부해야 함
inp = keras.Input((5, 3))
h = L.Masking(0.0)(inp)
for name, model in (("LSTM", seq(L.Masking(0.0), L.LSTM(4), L.Dense(2), shape=(5, 3))),
                    ("셀프 어텐션", keras.Model(inp, L.MultiHeadAttention(2, 4)(h, h)))):  # 어텐션은 예전에 조용히 통과했음
    with tempfile.TemporaryDirectory() as d:
        try:
            export(model, os.path.join(d, "m.knx"))
            print(f"  ✗ Masking → {name} 거부: 내보내기가 성공함")
            fail += 1
        except ValueError as e:
            ok = "Masking은 지원하지 않음" in str(e)
            print(f"  {'✓' if ok else '✗'} Masking → {name} 거부: {e}")
            fail += not ok

print("통과 ✓" if fail == 0 else f"실패 {fail}개 ✗")
sys.exit(fail != 0)
