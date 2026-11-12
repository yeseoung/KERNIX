"""Keras 3 모델(Sequential, 함수형, 중첩 모델) → KNX3 그래프 (tools/knx3.py).
export_keras.py의 export()가 일렬 LSTM/Dense 모델이 아닐 때 부른다. 지원하지 않는 레이어·설정은 ValueError로 거부한다.
"""
import numpy as np

import keras

from knx3 import FIRST, LAST, Graph

ACTS = {"relu": "relu", "gelu": "gelu", "silu": "silu", "swish": "silu", "sigmoid": "sigmoid", "tanh": "tanh",
        "elu": "elu", "selu": "selu", "softplus": "softplus", "relu6": "relu6", "hard_sigmoid": "hard_sigmoid",
        "hard_silu": "hard_swish", "hard_swish": "hard_swish", "exponential": "exp", "mish": "mish",
        "softsign": "softsign", "leaky_relu": "leaky_relu", "linear": None, "softmax": "softmax"}
ALPHA = {"leaky_relu": 0.2, "elu": 1.0}  # 활성화 함수 기본 인자 (keras.activations)
IDENTITY = {"Dropout", "SpatialDropout1D", "SpatialDropout2D", "GaussianNoise", "GaussianDropout", "AlphaDropout",
            "ActivityRegularization", "InputLayer", "Masking"}


def _np(v):
    return np.asarray(keras.ops.convert_to_numpy(v), np.float64)


def _act(fn):
    """activation 설정(함수 또는 이름) → (KERNIX 이름, alpha)"""
    name = fn if isinstance(fn, str) else keras.activations.serialize(fn)
    if isinstance(name, dict):
        name = name.get("config", {}).get("name") if isinstance(name.get("config"), dict) else name.get("config")
    if name not in ACTS:
        raise ValueError(f"지원하지 않는 활성화: {name}")
    return ACTS[name], ALPHA.get(name, 0.0)


def _same(n, k, s, d=1):
    """Keras/TF 'same' 패딩 (앞, 뒤): 남는 한 칸은 뒤에"""
    out = -(-n // s)
    total = max((out - 1) * s + (k - 1) * d + 1 - n, 0)
    return total // 2, total - total // 2


def _pads(cfg, shape, k, s, d):
    """(위, 아래, 왼, 오른)"""
    p = cfg["padding"]
    if p == "valid":
        return (0, 0, 0, 0)
    if p == "causal":
        return (d[0] * (k[0] - 1), 0, 0, 0)
    if p != "same":
        raise ValueError(f"지원하지 않는 padding: {p}")
    H, W = shape[0], (shape[1] if len(shape) == 3 else 1)
    return _same(H, k[0], s[0], d[0]) + _same(W, k[1], s[1], d[1])


def _geom(cfg, nd, key_k, key_s="strides", key_d="dilation_rate"):
    """1차원(nd 2)은 (k, 1) 꼴로"""
    def two(v, default=1):
        v = (v,) if isinstance(v, int) else tuple(v) if v is not None else (default,)
        return v + (1,) if len(v) == 1 else v
    k = two(cfg[key_k])
    s = two(cfg.get(key_s)) if cfg.get(key_s) is not None else k
    d = two(cfg.get(key_d, 1))
    if nd == 2 and (k[1] != 1 or s[1] != 1 or d[1] != 1):
        raise ValueError("1차원 레이어 설정이 잘못됨")
    return k, s, d


def _check_cl(cfg):
    if cfg.get("data_format", "channels_last") not in (None, "channels_last"):
        raise ValueError("data_format='channels_first'는 지원하지 않음 (channels_last로 학습한 모델만)")


def _rnn_weights(layer, kind):
    cfg = layer.get_config()
    if cfg.get("go_backwards") and not getattr(layer, "_kernix_backward", False):
        raise ValueError(f"{kind} go_backwards=True는 Bidirectional 안에서만 지원")
    if cfg.get("stateful") or cfg.get("return_state"):
        raise ValueError(f"{kind} stateful/return_state는 지원하지 않음")
    if kind in ("LSTM", "GRU") and cfg.get("recurrent_activation", "sigmoid") != "sigmoid":
        raise ValueError(f"{kind} recurrent_activation은 sigmoid만 지원")
    if kind == "GRU" and not cfg.get("reset_after", True):
        raise ValueError("GRU reset_after=False는 지원하지 않음")
    act, _ = _act(cfg["activation"])
    if act in (None, "softmax"):
        raise ValueError(f"{kind} activation={cfg['activation']}는 지원하지 않음")
    ws = [_np(w) for w in layer.get_weights()]
    w, u = ws[0], ws[1]
    gu = w.shape[1]
    if len(ws) > 2:
        b = ws[2].reshape(-1)
    else:
        b = np.zeros(2 * gu if kind == "GRU" else gu)
    return (w, u, b), act, cfg["units"], cfg["return_sequences"]


CELL = {"LSTM": "LSTM", "GRU": "GRU", "SimpleRNN": "RNN"}


def _layer(g, layer, node, xs):
    """레이어 호출 하나 → 출력 값 번호 목록"""
    kind, cfg = type(layer).__name__, layer.get_config()
    x = xs[0] if xs else None
    s = g.shape(x) if x is not None else None
    if kind in IDENTITY:
        return [x]
    if isinstance(layer, keras.Model):  # 중첩된 모델 (Sequential 안의 Sequential, 백본 등)
        return _subgraph(g, layer, xs)
    if kind == "Dense" or (kind == "TimeDistributed" and type(layer.layer).__name__ == "Dense"):
        inner = layer.layer if kind == "TimeDistributed" else layer
        c = inner.get_config()
        w = _np(inner.kernel)
        b = _np(inner.bias) if c["use_bias"] else None
        act, alpha = _act(c["activation"])
        return [g.act(g.linear(x, w, b), act, alpha)]
    if kind == "Activation":
        act, alpha = _act(cfg["activation"])
        return [g.act(x, act, alpha)]
    if kind == "ReLU":
        mv, ns, th = cfg.get("max_value"), cfg.get("negative_slope", 0.0), cfg.get("threshold", 0.0)
        if th:
            raise ValueError("ReLU threshold는 지원하지 않음")
        if mv is None:
            return [g.act(x, "leaky_relu", ns) if ns else g.act(x, "relu")]
        if float(mv) == 6.0 and not ns:
            return [g.act(x, "relu6")]
        raise ValueError(f"ReLU max_value={mv}는 지원하지 않음 (6만)")
    if kind == "LeakyReLU":
        return [g.act(x, "leaky_relu", cfg.get("negative_slope", cfg.get("alpha", 0.3)))]
    if kind == "ELU":
        return [g.act(x, "elu", cfg.get("alpha", 1.0))]
    if kind == "Softmax":
        if cfg.get("axis", -1) not in (-1, len(s)):
            raise ValueError("Softmax는 마지막 축만 지원")
        return [g.softmax(x)]
    if kind == "PReLU":  # relu(x) − alpha·relu(−x), alpha는 채널마다
        alpha = _np(layer.alpha)
        a = alpha.reshape(-1, alpha.shape[-1])
        if not np.allclose(a, a[:1]):
            raise ValueError("PReLU alpha가 채널 밖의 축에서 다름 (shared_axes로 채널만 남긴 경우만 지원)")
        neg = g.affine(g.act(g.affine(x, -1.0, 0.0), "relu"), a[0], 0.0)
        return [g.binary(g.act(x, "relu"), neg, "sub")]
    if kind == "Normalization":
        if cfg.get("invert"):
            raise ValueError("Normalization invert=True는 지원하지 않음")
        mean, var = _np(layer.mean).reshape(-1), _np(layer.variance).reshape(-1)
        std = np.maximum(np.sqrt(var), keras.backend.epsilon())
        return [g.affine(x, 1.0 / std, -mean / std)]
    if kind == "Rescaling":
        return [g.affine(x, np.asarray(cfg["scale"], np.float64).reshape(-1), np.asarray(cfg["offset"], np.float64).reshape(-1))]
    if kind == "BatchNormalization":
        if cfg["axis"] not in (-1, len(s), [-1], [len(s)]):
            raise ValueError("BatchNormalization은 마지막(채널) 축만 지원")
        c = s[-1]
        gamma = _np(layer.gamma) if cfg["scale"] else np.ones(c)
        beta = _np(layer.beta) if cfg["center"] else np.zeros(c)
        scale = gamma / np.sqrt(_np(layer.moving_variance) + cfg["epsilon"])
        return [g.affine(x, scale, beta - _np(layer.moving_mean) * scale)]
    if kind == "LayerNormalization":
        ax = cfg["axis"]
        if ax not in (-1, len(s), [-1], [len(s)]) or cfg.get("rms_scaling"):
            raise ValueError("LayerNormalization은 마지막 축만 지원 (rms_scaling 없음)")
        return [g.layernorm(x, _np(layer.gamma) if cfg["scale"] else None, _np(layer.beta) if cfg["center"] else None, cfg["epsilon"])]
    if kind in ("Conv1D", "Conv2D"):
        _check_cl(cfg)
        k, st, d = _geom(cfg, len(s), "kernel_size")
        w = _np(layer.kernel)
        cout = w.shape[-1]
        act, alpha = _act(cfg["activation"])
        y = g.conv(x, w.reshape(-1, cout), _np(layer.bias) if cfg["use_bias"] else None, k, st, d,
                   _pads(cfg, s, k, st, d), cfg.get("groups", 1))
        return [g.act(y, act, alpha)]
    if kind in ("DepthwiseConv1D", "DepthwiseConv2D", "SeparableConv1D", "SeparableConv2D"):
        _check_cl(cfg)
        k, st, d = _geom(cfg, len(s), "kernel_size")
        dw = _np(layer.depthwise_kernel if kind.startswith("Sep") else layer.kernel)
        cin, mult = dw.shape[-2], dw.shape[-1]
        sep = kind.startswith("Sep")
        y = g.conv(x, dw.reshape(-1, cin * mult), None if sep or not cfg["use_bias"] else _np(layer.bias), k, st, d,
                   _pads(cfg, s, k, st, d), cin)
        if sep:
            pw = _np(layer.pointwise_kernel)
            y = g.conv(y, pw.reshape(-1, pw.shape[-1]), _np(layer.bias) if cfg["use_bias"] else None, (1, 1))
        act, alpha = _act(cfg["activation"])
        return [g.act(y, act, alpha)]
    if kind in ("MaxPooling1D", "MaxPooling2D", "AveragePooling1D", "AveragePooling2D"):
        _check_cl(cfg)
        k, st, _ = _geom(cfg, len(s), "pool_size", "strides", "_none")
        return [g.pool(x, "max" if kind.startswith("Max") else "avg", k, st, _pads(cfg, s, k, st, (1, 1)), count_pad=False)]
    if kind in ("GlobalAveragePooling1D", "GlobalAveragePooling2D", "GlobalMaxPooling1D", "GlobalMaxPooling2D"):
        _check_cl(cfg)
        y = g.global_pool(x, "max" if "Max" in kind else "avg")
        return [g.reshape(y, (1,) * (len(s) - 1) + (s[-1],)) if cfg.get("keepdims") else y]
    if kind == "Flatten":
        return [g.reshape(x, (int(np.prod(s)),))]
    if kind == "Reshape":
        return [g.reshape(x, np.empty(s).reshape(cfg["target_shape"]).shape)]
    if kind == "Permute":
        return [g.permute(x, [d - 1 for d in cfg["dims"]] + ([len(s) - 1] if len(cfg["dims"]) < len(s) else []))]
    if kind in ("Add", "Subtract", "Multiply", "Maximum", "Minimum", "Average"):
        op = {"Add": "add", "Subtract": "sub", "Multiply": "mul", "Maximum": "max", "Minimum": "min", "Average": "add"}[kind]
        y = xs[0]
        for v in xs[1:]:
            y = g.binary(y, v, op)
        return [g.affine(y, 1.0 / len(xs), 0.0) if kind == "Average" else y]
    if kind == "Concatenate":
        ax = cfg["axis"] % (len(s) + 1)  # 배치 축 포함 번호
        if ax == len(s):
            return [g.concat(xs, LAST)]
        if ax == 1:
            return [g.concat(xs, FIRST)]
        raise ValueError("Concatenate는 첫 축(axis=1)이나 마지막 축만 지원")
    if kind == "Embedding":
        if cfg.get("mask_zero"):
            raise ValueError("Embedding mask_zero=True는 지원하지 않음")
        return [g.embedding(x, _np(layer.embeddings))]
    if kind in CELL:
        dw, act, units, ret = _rnn_weights(layer, kind)
        return [g.rnn(x, CELL[kind], units, [dw], act, ret)]
    if kind == "Bidirectional":
        f, b = layer.forward_layer, layer.backward_layer
        fk = type(f).__name__
        if fk not in CELL or cfg.get("merge_mode") not in ("concat", "sum", "mul", "ave"):
            raise ValueError(f"Bidirectional({fk}, merge_mode={cfg.get('merge_mode')})는 지원하지 않음")
        b._kernix_backward = True
        fw, act, units, ret = _rnn_weights(f, fk)
        bw, _, _, _ = _rnn_weights(b, fk)
        return [g.rnn(x, CELL[fk], units, [fw, bw], act, ret, cfg["merge_mode"])]
    if kind == "MultiHeadAttention":
        kw = node.arguments.kwargs
        axes = cfg.get("attention_axes")  # 빌드되면 기본값이 (1, ..., rank−2)로 채워짐
        if kw.get("attention_mask") is not None or kw.get("return_attention_scores") or (
                axes is not None and tuple(axes) != tuple(range(1, len(s)))):
            raise ValueError("MultiHeadAttention은 attention_mask, return_attention_scores, attention_axes 없이만 지원")
        args = list(node.arguments.args)
        q_t = args[0]
        v_t = args[1] if len(args) > 1 else kw.get("value")
        k_t = args[2] if len(args) > 2 else kw.get("key")
        val = _layer.val
        q, v = val[id(q_t)], val[id(v_t)]
        k = val[id(k_t)] if k_t is not None else None
        h = cfg["num_heads"]

        def proj(dense):
            w = _np(dense.kernel)
            return w.reshape(w.shape[0], -1), (_np(dense.bias).reshape(-1) if cfg["use_bias"] else None)
        wq, bq = proj(layer._query_dense)
        wk, bk = proj(layer._key_dense)
        wv, bv = proj(layer._value_dense)
        wo = _np(layer._output_dense.kernel)
        wo = wo.reshape(-1, wo.shape[-1])
        bo = _np(layer._output_dense.bias) if cfg["use_bias"] else None
        return [g.attention(q, h, wq, bq, wk, bk, wv, bv, wo, bo, v=None if v == q and k is None else v, k=k,
                            causal=bool(kw.get("use_causal_mask")))]
    if kind in ("ZeroPadding1D", "ZeroPadding2D"):
        _check_cl(cfg)
        p = cfg["padding"]
        if kind == "ZeroPadding1D":
            p = (p, p) if isinstance(p, int) else tuple(p)
            return [g.pad(x, (p[0], p[1], 0, 0))]
        (t, bt), (lf, rt) = [(v, v) if isinstance(v, int) else tuple(v) for v in p]
        return [g.pad(x, (t, bt, lf, rt))]
    if kind in ("UpSampling1D", "UpSampling2D"):
        if kind == "UpSampling2D":
            _check_cl(cfg)
            if cfg.get("interpolation", "nearest") != "nearest":
                raise ValueError("UpSampling2D는 nearest만 지원")
            return [g.upsample(x, *cfg["size"])]
        return [g.upsample(x, cfg["size"])]
    if kind in ("Cropping1D", "Cropping2D"):
        c = cfg["cropping"]
        if kind == "Cropping1D":
            t, bt = (c, c) if isinstance(c, int) else tuple(c)
            return [g.select(x, FIRST, t, s[0] - t - bt)]
        _check_cl(cfg)
        (t, bt), (lf, rt) = [(v, v) if isinstance(v, int) else tuple(v) for v in c]
        y = g.select(x, FIRST, t, s[0] - t - bt)  # H 방향
        H = g.shape(y)[0]
        y = g.reshape(y, (H, s[1] * s[2]))  # W 방향: (H, W·C)에서 열 범위
        y = g.select(y, LAST, lf * s[2], (s[1] - lf - rt) * s[2])
        return [g.reshape(y, (H, s[1] - lf - rt, s[2]))]
    raise ValueError(f"지원하지 않는 레이어: {kind}")


def _subgraph(g, model, in_vals):
    """model의 입력을 in_vals에 묶고 레이어를 위상 순서로 처리 → 출력 값 번호 목록"""
    if len(model.inputs) != len(in_vals):
        raise ValueError("중첩 모델의 입력 수가 맞지 않음")
    val = _layer.val
    for t, v in zip(model.inputs, in_vals):
        val[id(t)] = v
    # 출력까지 이어지는 노드만: 옛 .h5를 Keras 3에서 열면 같은 입력을 받는 쓰이지 않는 노드가 레이어마다 생김
    made_by = {id(t): (l, n) for l in model.layers for n in l._inbound_nodes for t in n.output_tensors}
    need, stack = set(), [id(t) for t in model.outputs]
    while stack:
        t = stack.pop()
        if t in made_by and id(made_by[t][1]) not in need:
            need.add(id(made_by[t][1]))
            stack += [id(x) for x in made_by[t][1].input_tensors]
    todo = [(l, n) for l in model.layers for n in l._inbound_nodes if id(n) in need]
    while True:
        ready = [(l, n) for l, n in todo if n.input_tensors and all(id(t) in val for t in n.input_tensors)
                 and not all(id(t) in val for t in n.output_tensors)]
        if not ready:
            break
        for l, n in ready:
            outs = _layer(g, l, n, [val[id(t)] for t in n.input_tensors])
            for t, v in zip(n.output_tensors, outs):
                val[id(t)] = v
            todo.remove((l, n))
    missing = [t for t in model.outputs if id(t) not in val]
    if missing:
        raise ValueError("입력에서 출력까지 이어지지 않는 그래프 (지원하지 않는 연결)")
    return [val[id(t)] for t in model.outputs]


def export_graph(model, path, quantize=None):
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("입력과 출력이 하나씩인 모델만 지원")
    in_shape = tuple(model.inputs[0].shape[1:])
    if any(d is None for d in in_shape):
        raise ValueError(f"입력 모양이 정해지지 않음 {in_shape}: keras.Input(shape=...)에 크기를 모두 지정")
    g = Graph(in_shape, 0, quantize)
    _layer.val = {}
    out, = _subgraph(g, model, [0])
    g.write(path, out)
