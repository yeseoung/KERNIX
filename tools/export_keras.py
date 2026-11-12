"""Keras 모델 → .knx. 일렬 LSTM/Dense 모델(수어 모델)은 KNX1/KNX2(export_sequence), 그 밖은 KNX3(keras_graph.py).
명령줄은 tools/convert.py"""
import os
import struct

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from knx_quant import MAGIC, weight_block  # noqa: E402

ACTS = {"relu": 0, "sigmoid": 3, "tanh": 4}


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def export(model, path, quantize=None):
    """일렬 LSTM/Dense 모델(수어 모델)은 KNX1/KNX2(export_sequence), 그 밖(합성곱, 잔차·어텐션 그래프, GRU 등)은
    KNX3(keras_graph.export_graph)로 내보낸다"""
    try:
        export_sequence(model, path, quantize)
    except ValueError as seq_err:
        from keras_graph import export_graph
        try:
            export_graph(model, path, quantize)
        except ValueError as e:
            raise ValueError(f"{e} (일렬 형식으로도 안 됨: {seq_err})") from None


def export_sequence(model, path, quantize=None):
    """Dropout은 추론에서 항등이라 건너뛰고, Normalization은 바로 다음 층의 가중치에 접어 넣는다.
    quantize="int8"이면 KNX2로: LSTM 입력 가중치는 I8_ROW(입력 특징별 scale, 접어 넣은 Normalization의 특징별 크기 차이를
    흡수, 로드할 때 float로 풂), LSTM 순환과 Dense는 I8_COL(출력 채널별, int8로 계산). docs/int8-quantization-design.md"""
    if quantize not in MAGIC:
        raise ValueError(f"지원하지 않는 quantize: {quantize}")
    v2 = quantize is not None
    blk = lambda w, mode: weight_block(w, mode if v2 else "f32", v2)  # noqa: E731
    if len(model.inputs) != 1 or len(model.outputs) != 1 or len(model.input_shape) not in (2, 3):
        raise ValueError("입력·출력이 하나인 (batch, F) 또는 (batch, T, F) 모델이 아님")
    # 일렬 연결만: 출력에서 입력까지 거슬러 올라간 사슬이 model.layers 순서와 같아야 함.
    # 앞에서부터 따라가면 안 됨: 옛 Keras로 저장한 .h5를 Keras 3에서 열면 레이어마다 같은 입력의 노드가 여러 개 생기고,
    # 그중 마지막 사슬만 모델 출력으로 이어짐
    made_by = {id(t): (layer, n) for layer in model.layers for n in layer._inbound_nodes for t in n.output_tensors}
    chain, cur = [], id(model.outputs[0])
    while cur != id(model.inputs[0]):
        if cur not in made_by or len(made_by[cur][1].input_tensors) != 1:
            raise ValueError("일렬 모델이 아님")
        chain.append(made_by[cur][0])
        cur = id(made_by[cur][1].input_tensors[0])
    if chain[::-1] != [layer for layer in model.layers if type(layer).__name__ != "InputLayer"]:
        raise ValueError("일렬 모델이 아님")
    seq_len = model.input_shape[1]
    ops, norm = [], None

    def fold(w, b):  # ((x - mean) / std) @ w + b  ==  x @ (w / std) + (b - (mean / std) @ w)
        nonlocal norm
        if norm is not None:
            mean, std = norm
            b = b - (mean / std) @ w
            w = w / std[:, None]
            norm = None
        return w, b

    for layer in model.layers:
        kind, cfg = type(layer).__name__, layer.get_config()
        if kind in ("Dropout", "InputLayer"):
            continue
        if kind == "Normalization":
            if cfg.get("invert") or norm is not None:
                raise ValueError("지원하지 않는 Normalization 설정")
            std = np.maximum(np.sqrt(np.array(layer.variance, np.float64).reshape(-1)), tf.keras.backend.epsilon())
            norm = (np.array(layer.mean, np.float64).reshape(-1), std)
        elif kind == "LSTM":
            if len(model.input_shape) != 3:
                raise ValueError("LSTM 입력은 (batch, T, F)여야 함")
            if (cfg["recurrent_activation"] != "sigmoid" or cfg["activation"] not in ("relu", "tanh") or not cfg["use_bias"]
                    or cfg["go_backwards"] or cfg["stateful"] or cfg["return_state"]):
                raise ValueError(f"지원하지 않는 LSTM 설정: {cfg}")
            w, u, b = (a.astype(np.float64) for a in layer.get_weights())  # (in, 4u), (u, 4u), (4u,), 게이트 순서 i f c o
            w, b = fold(w, b)
            ops.append(struct.pack("<6I", 4, w.shape[0], cfg["units"], ACTS[cfg["activation"]],
                                   cfg["return_sequences"], seq_len) + blk(w, "i8_row") + blk(u, "i8_col") + f32(b))
        elif kind == "Dense":
            w, b = (a.astype(np.float64) for a in layer.get_weights()) if cfg["use_bias"] else (layer.get_weights()[0].astype(np.float64), None)
            w, b = fold(w, b if b is not None else np.zeros(w.shape[1]))
            ops.append(struct.pack("<4I", 0, w.shape[0], w.shape[1], 1) + blk(w, "i8_col") + f32(b))
            if cfg["activation"] == "softmax":
                ops.append(struct.pack("<I", 3))
            elif cfg["activation"] in ACTS:
                ops.append(struct.pack("<2I", 1, ACTS[cfg["activation"]]))
            elif cfg["activation"] != "linear":
                raise ValueError(f"지원하지 않는 활성화: {cfg['activation']}")
        else:
            raise ValueError(f"지원하지 않는 레이어: {kind}")
    if norm is not None:
        raise ValueError("Normalization 뒤에 가중치 층이 없음")
    if len(model.input_shape) == 3 and not any(type(layer).__name__ == "LSTM" for layer in model.layers):
        raise ValueError("(batch, T, F) 입력인데 LSTM이 없음: 시점별 출력 모양은 KNX3로")
    with open(path, "wb") as f:
        f.write(MAGIC[quantize] + struct.pack("<I", len(ops)) + b"".join(ops))
