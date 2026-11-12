"""PyTorch 모델 → KNX3 그래프 (tools/knx3.py). torch.fx로 그래프를 뽑고, 예제 입력으로 한 번 실행해 노드마다 실제 모양을 기록한다.

KERNIX는 channels-last로 저장하고 PyTorch는 channels-first(N, C, H, W)라서, 값마다 "논리 축 i(PyTorch, 배치 제외)가
저장 축 lperm[i]에 있다"를 들고 다닌다. permute/transpose는 이 대응만 바꾸고, 연산이 특정 배치를 요구할 때만 PERMUTE를 넣는다.
입력과 무관한 노드(파라미터, arange, size() 등)는 기록한 값을 상수로 쓴다. 지원하지 않는 것은 ValueError로 거부한다.
"""
import math
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx

from knx3 import FIRST, LAST, Graph

LEAF = (nn.LSTM, nn.GRU, nn.RNN, nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerEncoder)
CF_FIRST = (nn.Conv1d, nn.Conv2d, nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d, nn.BatchNorm2d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)
ACT_MOD = {nn.ReLU: "relu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh", nn.SiLU: "silu", nn.SELU: "selu", nn.ReLU6: "relu6",
           nn.Hardsigmoid: "hard_sigmoid", nn.Hardswish: "hard_swish", nn.Mish: "mish", nn.Softsign: "softsign"}
ACT_FN = {F.relu: "relu", torch.relu: "relu", torch.sigmoid: "sigmoid", F.sigmoid: "sigmoid", torch.tanh: "tanh",
          F.tanh: "tanh", F.silu: "silu", F.selu: "selu", F.relu6: "relu6", F.hardsigmoid: "hard_sigmoid",
          F.hardswish: "hard_swish", F.mish: "mish", F.softsign: "softsign", torch.exp: "exp"}
ACT_METHOD = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "exp": "exp"}
BIN_FN = {operator.add: "add", torch.add: "add", operator.sub: "sub", torch.sub: "sub", operator.mul: "mul",
          torch.mul: "mul", operator.truediv: "div", torch.div: "div", torch.maximum: "max", torch.minimum: "min"}
BIN_METHOD = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}
IDENT_METHOD = {"contiguous", "clone", "detach", "float", "to"}


def _np(t):
    return t.detach().cpu().double().numpy()


def _ident(n):
    return tuple(range(n))


def _cf(n):
    """channels-first 논리 (C, s1, s2, ...)를 channels-last (s1, s2, ..., C)로 저장할 때의 대응"""
    return (n - 1,) + tuple(range(n - 1)) if n >= 2 else (0,)


class T:
    """KERNIX 값 v와 논리 축 대응 lperm (논리 축 i = 저장 축 lperm[i])"""
    def __init__(self, v, lperm):
        self.v, self.lperm = v, tuple(lperm)


class Seq:
    """RNN 출력 (output, h_n): output만 KERNIX 값, h_n은 output에서 고름"""
    def __init__(self, out, units, dirs, cell, layers):
        self.out, self.units, self.dirs, self.cell, self.layers = out, units, dirs, cell, layers


class HN:
    def __init__(self, seq):
        self.seq = seq


class Pair:  # MultiheadAttention (출력, 가중치)
    def __init__(self, out):
        self.out = out


class _Tracer(fx.Tracer):
    def is_leaf_module(self, m, qualname):
        return isinstance(m, LEAF) or super().is_leaf_module(m, qualname)


class _Rec(fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.vals = {}

    def run_node(self, n):
        r = super().run_node(n)
        self.vals[n] = r
        return r


class Exporter:
    def __init__(self, model, input_shape, quantize):
        self.model = model.eval()
        self.gm = fx.GraphModule(model, _Tracer().trace(model))
        self.nodes = list(self.gm.graph.nodes)
        ph = [n for n in self.nodes if n.op == "placeholder"]
        if len(ph) != 1:
            raise ValueError("입력이 하나인 모델만 지원")
        self.ph = ph[0]
        first = self._first_consumer(self.ph)
        if isinstance(first, nn.Embedding):
            ex = torch.randint(0, first.num_embeddings, (2,) + tuple(input_shape))
        else:
            ex = torch.randn((2,) + tuple(input_shape))
        rec = _Rec(self.gm)
        with torch.no_grad():
            rec.run(ex)
        self.rec = rec.vals
        nd = len(input_shape)
        self.layout = 1 if nd >= 2 and isinstance(first, CF_FIRST) else 0
        lp = _cf(nd) if self.layout else _ident(nd)
        store = [0] * nd
        for i, j in enumerate(lp):
            store[j] = input_shape[i]
        self.g = Graph(store, self.layout, quantize)
        self.env = {self.ph: T(0, lp)}
        self.dep = {self.ph}

    def _first_consumer(self, node):
        seen, todo = set(), list(node.users)
        while todo:
            n = todo.pop(0)
            if n in seen:
                continue
            seen.add(n)
            if n.op == "call_module":
                m = self.gm.get_submodule(n.target)
                if not isinstance(m, (nn.Dropout, nn.Identity)):
                    return m
            if n.op in ("call_method", "call_function") and (n.target in IDENT_METHOD or n.target is F.dropout):
                todo += list(n.users)
                continue
            return None
        return None

    # ---- 배치 맞추기 ----
    def shape(self, t):
        """논리 모양 (배치 제외)"""
        s = self.g.shape(t.v)
        return tuple(s[j] for j in t.lperm)

    def layout_to(self, t, want):
        """논리 축 i가 저장 축 want[i]에 오도록 PERMUTE"""
        want = tuple(want)
        if t.lperm == want:
            return t
        inv_want = [0] * len(want)
        for i, j in enumerate(want):
            inv_want[j] = i
        perm = [t.lperm[inv_want[j]] for j in range(len(want))]
        return T(self.g.permute(t.v, perm), want)

    def last_is_channel(self, t, axis=-1):
        """논리 축 axis가 저장 마지막 축에 있게 (Linear, LayerNorm, softmax): 다른 축 순서는 그대로 둬도 됨"""
        n = len(t.lperm)
        axis %= n
        if t.lperm[axis] == n - 1:
            return t
        want = list(t.lperm)
        j = want.index(n - 1)
        want[axis], want[j] = n - 1, want[axis]
        return self.layout_to(t, want)

    # ---- 노드 처리 ----
    def arg(self, a):
        if isinstance(a, fx.Node):
            return self.env[a] if a in self.dep else self.rec[a]
        if isinstance(a, (list, tuple)):
            return type(a)(self.arg(x) for x in a)
        return a

    def run(self):
        out = None
        live, todo = set(), [n for n in self.nodes if n.op == "output"]  # 출력에 쓰이는 노드만 (튜플 풀기의 _ 등은 건너뜀)
        while todo:
            n = todo.pop()
            if n not in live:
                live.add(n)
                todo += n.all_input_nodes
        for n in self.nodes:
            if n.op == "placeholder" or n not in live:
                continue
            if n.op == "output":
                out = self.arg(n.args[0])
                break
            deps = [a for a in n.all_input_nodes if a in self.dep]
            if not deps or isinstance(self.rec[n], (int, float, bool, torch.Size)) or n.target in ("size", "dim"):
                continue  # 입력과 무관하거나 텐서가 아닌 값(크기 등): 기록한 값이 상수
            self.dep.add(n)
            self.env[n] = self.node(n)
        if not isinstance(out, T):
            raise ValueError("출력이 텐서 하나인 모델만 지원")
        nd = len(out.lperm)
        out = self.layout_to(out, _cf(nd) if self.layout and nd >= 2 else _ident(nd))
        return out.v

    def node(self, n):
        args = [self.arg(a) for a in n.args]
        kw = {k: self.arg(v) for k, v in n.kwargs.items()}
        rec = self.rec[n]
        if n.op == "call_module":
            return self.module(self.gm.get_submodule(n.target), args, kw, rec)
        if n.op == "call_function":
            return self.function(n.target, args, kw, rec)
        if n.op == "call_method":
            return self.method(n.target, args, kw, rec)
        raise ValueError(f"지원하지 않는 노드: {n.op} {n.target}")

    def act(self, t, name, alpha=0.0):
        return T(self.g.act(t.v, name, alpha), t.lperm)

    def unary_act(self, t, name, kw, alpha_key=None, default=0.0):
        return self.act(t, name, float(kw.get(alpha_key, default)) if alpha_key else 0.0)

    def softmax(self, t, dim):
        t = self.last_is_channel(t, dim - 1 if dim > 0 else dim)
        return T(self.g.softmax(t.v), t.lperm)

    def reshape(self, t, rec):
        t = self.layout_to(t, _ident(len(t.lperm)))
        shape = tuple(rec.shape[1:])
        if not shape:
            raise ValueError("배치 축만 남는 reshape는 지원하지 않음")
        return T(self.g.reshape(t.v, shape), _ident(len(shape)))

    def binary(self, a, b, op, rec):
        if not isinstance(b, T):
            if not isinstance(a, T):
                raise ValueError("상수끼리 연산")
            return self.const_binary(a, b, op, rec, right=True)
        if not isinstance(a, T):
            return self.const_binary(b, a, op, rec, right=False)
        sa, sb = self.shape(a), self.shape(b)
        if len(sa) < len(sb):  # 뒤 축 기준 브로드캐스트: 짧은 쪽 앞에 1을 채움
            a = self.reshape(a, torch.empty((1,) + (1,) * (len(sb) - len(sa)) + sa))
        elif len(sb) < len(sa):
            b = self.reshape(b, torch.empty((1,) + (1,) * (len(sa) - len(sb)) + sb))
        big = a if np.prod(self.shape(a)) >= np.prod(self.shape(b)) else b
        a, b = self.layout_to(a, big.lperm), self.layout_to(b, big.lperm)
        return T(self.g.binary(a.v, b.v, op), big.lperm)

    def const_binary(self, t, c, op, rec, right):
        """t (op) c 또는 c (op) t, c는 스칼라나 상수 텐서"""
        s = self.shape(t)
        nd = len(s)
        if isinstance(c, (int, float)) or (torch.is_tensor(c) and c.numel() == 1):
            k = float(c)
            if op == "add":
                return T(self.g.affine(t.v, 1.0, k), t.lperm)
            if op == "mul":
                return T(self.g.affine(t.v, k, 0.0), t.lperm)
            if op == "sub":
                return T(self.g.affine(t.v, 1.0, -k) if right else self.g.affine(t.v, -1.0, k), t.lperm)
            if op == "div" and right:
                return T(self.g.affine(t.v, 1.0 / k, 0.0), t.lperm)
            raise ValueError(f"스칼라 {op}는 지원하지 않음")
        c = _np(c)
        if c.ndim == nd + 1 and c.shape[0] == 1:
            c = c[0]
        if c.ndim > nd:
            raise ValueError(f"상수 모양 {c.shape}이 입력 {s}보다 큼 (배치 축 방향 상수는 지원하지 않음)")
        c = c.reshape((1,) * (nd - c.ndim) + c.shape)
        chan = t.lperm.index(nd - 1)  # 저장 채널 축에 해당하는 논리 축
        per_channel = all(c.shape[i] == 1 for i in range(nd) if i != chan)
        if per_channel and op in ("add", "sub", "mul") or (per_channel and op == "div" and right):
            v = np.broadcast_to(c.reshape(-1) if c.shape[chan] > 1 else c.reshape(1), (s[chan],))
            if op == "add":
                return T(self.g.affine(t.v, 1.0, v), t.lperm)
            if op == "mul":
                return T(self.g.affine(t.v, v, 0.0), t.lperm)
            if op == "div":
                return T(self.g.affine(t.v, 1.0 / v, 0.0), t.lperm)
            return T(self.g.affine(t.v, 1.0, -v) if right else self.g.affine(t.v, -1.0, v), t.lperm)
        if op == "add" or (op == "sub" and right):
            full = np.broadcast_to(c, s) * (1.0 if op == "add" else -1.0)
            inv = [0] * nd
            for i, j in enumerate(t.lperm):
                inv[j] = i
            return T(self.g.add_const(t.v, np.transpose(full, inv)), t.lperm)  # 저장 순서로
        raise ValueError(f"상수 텐서와의 {op}는 지원하지 않음 (덧셈, 채널마다의 곱만)")

    def select(self, t, index, rec):
        """x[:, ...] 꼴: 배치 다음 축 하나만 정수 또는 슬라이스"""
        if not isinstance(index, tuple):
            index = (index,)
        if index[0] != slice(None):
            raise ValueError("배치 축을 자르는 인덱싱은 지원하지 않음")
        rest = list(index[1:])
        if Ellipsis in rest:
            e = rest.index(Ellipsis)
            rest = rest[:e] + [slice(None)] * (len(t.lperm) - len(rest) + 1) + rest[e + 1:]
        picks = [(i, r) for i, r in enumerate(rest) if r != slice(None)]
        if len(picks) != 1 or any(r is None for r in rest):
            raise ValueError(f"지원하지 않는 인덱싱: {index}")
        ax, r = picks[0]
        size = self.shape(t)[ax]
        if isinstance(r, int):
            start, length, squeeze = r % size, 1, True
        elif isinstance(r, slice) and r.step in (None, 1):
            start, stop, _ = r.indices(size)
            length, squeeze = stop - start, False
        else:
            raise ValueError(f"지원하지 않는 인덱싱: {index}")
        nd = len(t.lperm)
        j = t.lperm[ax]
        if j not in (0, nd - 1):  # 저장의 첫 축이나 마지막 축이어야 함
            t = self.layout_to(t, _ident(nd))
            j = ax
            if j not in (0, nd - 1):
                raise ValueError("가운데 축을 자르는 인덱싱은 지원하지 않음")
        axis = FIRST if j == 0 else LAST
        v = self.g.select(t.v, axis, start, length, squeeze and nd > 1)
        lp = list(t.lperm)
        if squeeze and nd > 1:
            del lp[ax]
            lp = [p - (p > j) for p in lp]
        return T(v, lp)

    def pool_global(self, t, kind, keep_shape=None):
        nd = len(t.lperm)
        t = self.layout_to(t, _cf(nd))
        y = T(self.g.global_pool(t.v, kind), (0,))
        if keep_shape is not None:  # (C, 1, 1) 등
            y = T(self.g.reshape(y.v, tuple(keep_shape[1:]) + (keep_shape[0],)), _cf(len(keep_shape)))
        return y

    def reduce(self, t, dims, keepdim, kind, rec):
        """남는 축이 하나(채널)인 평균/최댓값 = 전역 풀링"""
        nd = len(t.lperm)
        dims = {d % (nd + 1) - 1 for d in (dims if isinstance(dims, (list, tuple)) else [dims])}
        chan = [i for i in range(nd) if i not in dims]
        if -1 in dims or len(chan) != 1:
            raise ValueError("평균/최댓값은 축 하나만 남기는 경우만 지원")
        t = self.last_is_channel(t, chan[0])
        y = T(self.g.global_pool(t.v, kind), (0,))
        if keepdim:
            return self.from_rec_reshape(y, rec)
        return y

    def from_rec_reshape(self, y, rec):
        return T(self.g.reshape(y.v, tuple(rec.shape[1:])), _ident(rec.dim() - 1))

    # ---- 모듈 ----
    def module(self, m, args, kw, rec):
        x = args[0] if args else None
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout, nn.Identity)):
            return x
        if type(m) in ACT_MOD:
            return self.act(x, ACT_MOD[type(m)])
        if isinstance(m, nn.GELU):
            return self.act(x, "gelu_tanh" if m.approximate == "tanh" else "gelu")
        if isinstance(m, nn.LeakyReLU):
            return self.act(x, "leaky_relu", m.negative_slope)
        if isinstance(m, nn.ELU):
            return self.act(x, "elu", m.alpha)
        if isinstance(m, nn.Softplus):
            if m.beta != 1 or m.threshold != 20:
                raise ValueError("Softplus는 beta=1, threshold=20만 지원")
            return self.act(x, "softplus")
        if isinstance(m, nn.PReLU):
            a = _np(m.weight)
            nd = len(x.lperm)
            if a.size > 1:
                x = self.layout_to(x, _cf(nd))  # num_parameters = 채널 수 (논리 축 0)
            neg = self.g.affine(self.g.act(self.g.affine(x.v, -1.0, 0.0), "relu"), a, 0.0)
            return T(self.g.binary(self.g.act(x.v, "relu"), neg, "sub"), x.lperm)
        if isinstance(m, nn.Softmax):
            return self.softmax(x, m.dim if m.dim is not None else -1)
        if isinstance(m, nn.Linear):
            x = self.last_is_channel(x)
            return T(self.g.linear(x.v, _np(m.weight).T, _np(m.bias) if m.bias is not None else None), x.lperm)
        if isinstance(m, nn.LayerNorm):
            if len(m.normalized_shape) != 1:
                raise ValueError("LayerNorm은 마지막 축 하나만 지원")
            x = self.last_is_channel(x)
            w = _np(m.weight) if m.weight is not None else None
            b = _np(m.bias) if getattr(m, "bias", None) is not None else None
            return T(self.g.layernorm(x.v, w, b, m.eps), x.lperm)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nd = len(x.lperm)
            x = self.layout_to(x, _cf(nd))
            scale = (_np(m.weight) if m.affine else 1.0) / np.sqrt(_np(m.running_var) + m.eps)
            shift = (_np(m.bias) if m.affine else 0.0) - _np(m.running_mean) * scale
            return T(self.g.affine(x.v, scale, shift), x.lperm)
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            return self.conv(m, x)
        if isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d)):
            return self.pool(m, x)
        if isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d)):
            kind = "max" if "Max" in type(m).__name__ else "avg"
            out = m.output_size if isinstance(m.output_size, tuple) else (m.output_size,) * (len(x.lperm) - 1)
            if all(o == 1 for o in out):
                return self.pool_global(x, kind, tuple(rec.shape[1:]))
            sp = self.shape(x)[1:]
            if any(o is None or s % o for s, o in zip(sp, out)):
                raise ValueError("AdaptivePool은 출력 크기가 입력을 나누어떨어질 때만 지원")
            k = tuple(s // o for s, o in zip(sp, out)) + ((1,) if len(sp) == 1 else ())
            x = self.layout_to(x, _cf(len(x.lperm)))
            return T(self.g.pool(x.v, kind, k, k), x.lperm)
        if isinstance(m, nn.Flatten):
            return self.reshape(x, rec)
        if isinstance(m, nn.Unflatten):
            return self.reshape(x, rec)
        if isinstance(m, nn.Embedding):
            if m.max_norm is not None:
                raise ValueError("Embedding max_norm은 지원하지 않음")
            x = self.layout_to(x, _ident(len(x.lperm)))
            return T(self.g.embedding(x.v, _np(m.weight)), _ident(len(x.lperm) + 1))
        if isinstance(m, nn.Upsample):
            if m.mode != "nearest" or m.scale_factor is None:
                raise ValueError("Upsample은 nearest, scale_factor만 지원")
            return self.upsample(x, m.scale_factor)
        if isinstance(m, (nn.ZeroPad1d, nn.ZeroPad2d, nn.ConstantPad1d, nn.ConstantPad2d)):
            if getattr(m, "value", 0) != 0:
                raise ValueError("0이 아닌 값으로 채우는 패딩은 지원하지 않음")
            return self.pad(x, m.padding)
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            return self.rnn(m, x, args[1:], kw)
        if isinstance(m, nn.MultiheadAttention):
            return Pair(self.mha(m, args, kw))
        if isinstance(m, nn.TransformerEncoderLayer):
            return self.encoder_layer(m, x, self.causal_of(args[1:], kw, ("src_mask", "mask")))
        if isinstance(m, nn.TransformerEncoder):
            causal = self.causal_of(args[1:], kw, ("mask",))
            for layer in m.layers:
                x = self.encoder_layer(layer, x, causal)
            if m.norm is not None:
                x = self.module(m.norm, [x], {}, None)
            return x
        if isinstance(m, nn.Sequential):  # 리프로 남은 경우만 (보통 fx가 펼침)
            for sub in m:
                x = self.module(sub, [x], {}, None)
            return x
        raise ValueError(f"지원하지 않는 모듈: {type(m).__name__}")

    def conv(self, m, x):
        nd = len(x.lperm)
        x = self.layout_to(x, _cf(nd))
        if m.padding_mode != "zeros":
            raise ValueError("Conv padding_mode는 zeros만 지원")
        two = nd == 3
        w = _np(m.weight)  # (cout, cin/g, kh[, kw])
        if not two:
            w = w[..., None]
        k = w.shape[2:]
        st = tuple(m.stride) + (() if two else (1,))
        dil = tuple(m.dilation) + (() if two else (1,))
        if m.padding == "same":
            pads = []
            for kk, dd in zip(k, dil):
                tot = dd * (kk - 1)
                pads += [tot // 2, tot - tot // 2]
        elif m.padding == "valid":
            pads = [0, 0, 0, 0]
        else:
            p = tuple(m.padding) + (() if two else (0,))
            pads = [p[0], p[0], p[1], p[1]]
        wk = np.transpose(w, (2, 3, 1, 0)).reshape(-1, w.shape[0])  # (ky, kx, ci) × cout
        return T(self.g.conv(x.v, wk, _np(m.bias) if m.bias is not None else None, k, st, dil, pads, m.groups), x.lperm)

    def pool(self, m, x):
        nd = len(x.lperm)
        x = self.layout_to(x, _cf(nd))
        two = nd == 3

        def pair(v, fill):
            v = tuple(v) if isinstance(v, (tuple, list)) else (v,) * (2 if two else 1)
            return v if two else (v[0], fill)
        if getattr(m, "ceil_mode", False) or (hasattr(m, "dilation") and m.dilation not in (1, (1, 1))) or getattr(m, "divisor_override", None):
            raise ValueError("Pool의 ceil_mode, dilation, divisor_override는 지원하지 않음")
        k = pair(m.kernel_size, 1)
        st = pair(m.stride if m.stride is not None else m.kernel_size, 1)
        p = pair(m.padding, 0)
        pads = (p[0], p[0], p[1], p[1])
        avg = "Avg" in type(m).__name__
        return T(self.g.pool(x.v, "avg" if avg else "max", k, st, pads, count_pad=avg and m.count_include_pad), x.lperm)

    def upsample(self, x, scale):
        nd = len(x.lperm)
        x = self.layout_to(x, _cf(nd))
        sc = (scale,) * (nd - 1) if isinstance(scale, (int, float)) else tuple(scale)
        if any(float(s) != int(s) for s in sc):
            raise ValueError("Upsample은 정수 배율만 지원")
        sc = tuple(int(s) for s in sc)
        return T(self.g.upsample(x.v, sc[0], sc[1] if nd == 3 else 1), x.lperm)

    def pad(self, x, p):
        nd = len(x.lperm)
        x = self.layout_to(x, _cf(nd))
        p = tuple(p) if not isinstance(p, int) else (p,) * (2 * (nd - 1))
        if nd == 2:
            return T(self.g.pad(x.v, (p[0], p[1], 0, 0)), x.lperm)
        if len(p) != 4:
            raise ValueError("2차원 패딩은 (왼, 오른, 위, 아래)만 지원")
        return T(self.g.pad(x.v, (p[2], p[3], p[0], p[1])), x.lperm)

    def rnn(self, m, x, extra, kw):
        if not m.batch_first:
            raise ValueError(f"{type(m).__name__}은 batch_first=True만 지원")
        if any(e is not None for e in extra) or kw.get("hx") is not None:
            raise ValueError("RNN 초기 상태(hx) 입력은 지원하지 않음")
        if getattr(m, "proj_size", 0):
            raise ValueError("LSTM proj_size는 지원하지 않음")
        cell = {nn.LSTM: "LSTM", nn.GRU: "GRU", nn.RNN: "RNN"}[type(m)]
        act = "tanh" if cell != "RNN" else m.nonlinearity
        x = self.layout_to(x, _ident(len(x.lperm)))
        u = m.hidden_size
        dirs = 2 if m.bidirectional else 1
        v = x.v
        for layer in range(m.num_layers):
            dw = []
            for d in range(dirs):
                sfx = f"_l{layer}" + ("_reverse" if d else "")
                wi, wh = _np(getattr(m, "weight_ih" + sfx)), _np(getattr(m, "weight_hh" + sfx))
                bi = _np(getattr(m, "bias_ih" + sfx)) if m.bias else np.zeros(wi.shape[0])
                bh = _np(getattr(m, "bias_hh" + sfx)) if m.bias else np.zeros(wi.shape[0])
                if cell == "GRU":  # PyTorch r, z, n → Keras z, r, h
                    order = np.concatenate([np.arange(u, 2 * u), np.arange(u), np.arange(2 * u, 3 * u)])
                    dw.append((wi[order].T, wh[order].T, np.concatenate([bi[order], bh[order]])))
                else:
                    dw.append((wi.T, wh.T, bi + bh))
            v = self.g.rnn(v, cell, u, dw, act, ret_seq=True)
        return Seq(T(v, (0, 1)), u, dirs, cell, m.num_layers)

    def causal_of(self, args, kw, names):
        mask = args[0] if args else next((kw[k] for k in names if kw.get(k) is not None), None)
        if mask is None:
            if kw.get("is_causal"):
                raise ValueError("is_causal=True는 마스크 텐서와 같이 줘야 함")
            return False
        if isinstance(mask, T):
            raise ValueError("입력에 따라 바뀌는 마스크는 지원하지 않음")
        mk = mask.detach().cpu()
        n = mk.shape[-1]
        if mk.dim() != 2 or mk.shape[0] != n:
            raise ValueError("어텐션 마스크는 (L, L) 인과 마스크만 지원")
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        blocked = mk if mk.dtype == torch.bool else torch.isinf(mk) & (mk < 0)
        if not torch.equal(blocked, upper) or (mk.dtype != torch.bool and torch.any(mk[~upper] != 0)):
            raise ValueError("어텐션 마스크는 인과 마스크(위 삼각 −inf 또는 True)만 지원")
        return True

    def mha(self, m, args, kw, causal=None):
        if not m.batch_first:
            raise ValueError("MultiheadAttention은 batch_first=True만 지원")
        if m.bias_k is not None or m.add_zero_attn:
            raise ValueError("MultiheadAttention add_bias_kv/add_zero_attn은 지원하지 않음")
        # forward(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)
        a = list(args) + [None] * (6 - len(args))
        q = a[0]
        k = a[1] if a[1] is not None else kw.get("key")
        v = a[2] if a[2] is not None else kw.get("value")
        if (a[3] if a[3] is not None else kw.get("key_padding_mask")) is not None:
            raise ValueError("key_padding_mask는 지원하지 않음")
        if causal is None:
            causal = self.causal_of([a[5]] if a[5] is not None else [], kw, ("attn_mask",))
        E, H = m.embed_dim, m.num_heads
        if m._qkv_same_embed_dim:
            w = _np(m.in_proj_weight)
            wq, wk, wv = w[:E].T, w[E:2 * E].T, w[2 * E:].T
        else:
            wq, wk, wv = _np(m.q_proj_weight).T, _np(m.k_proj_weight).T, _np(m.v_proj_weight).T
        b = _np(m.in_proj_bias) if m.in_proj_bias is not None else None
        bq, bk, bv = (b[:E], b[E:2 * E], b[2 * E:]) if b is not None else (None, None, None)
        wo, bo = _np(m.out_proj.weight).T, (_np(m.out_proj.bias) if m.out_proj.bias is not None else None)
        if (b is None) != (bo is None):
            raise ValueError("MultiheadAttention bias는 입력 투영과 출력 투영에 같이 있거나 같이 없어야 함")
        q, k, v = (self.last_is_channel(t) for t in (q, k, v))
        q, k, v = q, self.layout_to(k, q.lperm), self.layout_to(v, q.lperm)
        same = q.v == k.v == v.v
        out = self.g.attention(q.v, H, wq, bq, wk, bk, wv, bv, wo, bo, v=None if same else v.v,
                               k=None if same or k.v == v.v else k.v, causal=causal)
        return T(out, q.lperm)

    def encoder_layer(self, m, x, causal):
        if not m.self_attn.batch_first:
            raise ValueError("TransformerEncoderLayer는 batch_first=True만 지원")
        a = m.activation
        act = "relu" if a in (F.relu, torch.relu) or isinstance(a, nn.ReLU) else "gelu" if a is F.gelu or (
            isinstance(a, nn.GELU) and a.approximate == "none") else None
        if act is None:
            raise ValueError("TransformerEncoderLayer activation은 relu/gelu만 지원")

        def sa(t):
            return self.mha(m.self_attn, [t, t, t], {}, causal)

        def ff(t):
            h = self.module(m.linear1, [t], {}, None)
            return self.module(m.linear2, [self.act(h, act)], {}, None)

        def norm(mod, t):
            return self.module(mod, [t], {}, None)

        add = lambda p, q: self.binary(p, q, "add", None)  # noqa: E731
        if m.norm_first:
            x = add(x, sa(norm(m.norm1, x)))
            return add(x, ff(norm(m.norm2, x)))
        x = norm(m.norm1, add(x, sa(x)))
        return norm(m.norm2, add(x, ff(x)))

    # ---- 함수 · 메서드 ----
    def getitem(self, x, idx, rec):
        if isinstance(x, Seq):
            if idx == 0:
                return x.out
            if idx == 1:
                return (HN(x), None) if x.cell == "LSTM" else HN(x)
            raise ValueError("RNN 출력 인덱스는 0, 1만 지원")
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], HN):  # LSTM (h_n, c_n)
            if idx == 0:
                return x[0]
            raise ValueError("LSTM c_n은 지원하지 않음")
        if isinstance(x, HN):  # h_n[i]: 마지막 층만 (출력 시퀀스에서 고름)
            s = x.seq
            T_len = self.g.shape(s.out.v)[0]
            if not isinstance(idx, int):
                raise ValueError("h_n은 정수 인덱스만 지원")
            i = idx if idx < 0 else idx - s.layers * s.dirs  # 마지막 층 기준 음수 번호로
            if s.dirs == 1 and i == -1:
                return self.select(s.out, (slice(None), T_len - 1), None)
            if s.dirs == 2 and i in (-1, -2):
                t = self.select(s.out, (slice(None), 0 if i == -1 else T_len - 1), None)
                return self.select(t, (slice(None), slice(s.units if i == -1 else 0, 2 * s.units if i == -1 else s.units)), None)
            raise ValueError("h_n은 마지막 층(h_n[-1], 양방향이면 h_n[-2]도)만 지원")
        if isinstance(x, Pair):
            if idx == 0:
                return x.out
            raise ValueError("어텐션 가중치 출력은 지원하지 않음")
        if isinstance(x, T):
            return self.select(x, idx, rec)
        raise ValueError(f"지원하지 않는 인덱싱 대상: {type(x).__name__}")

    def function(self, f, args, kw, rec):
        x = args[0] if args else None
        if f is operator.getitem:
            return self.getitem(args[0], args[1], rec)
        if f in BIN_FN:
            return self.binary(args[0], args[1], BIN_FN[f], rec)
        if f is operator.neg or f is torch.neg:
            return T(self.g.affine(x.v, -1.0, 0.0), x.lperm)
        if f in ACT_FN:
            return self.act(x, ACT_FN[f])
        if f is F.gelu:
            return self.act(x, "gelu_tanh" if kw.get("approximate", "none") == "tanh" else "gelu")
        if f is F.leaky_relu:
            return self.act(x, "leaky_relu", float(args[1] if len(args) > 1 else kw.get("negative_slope", 0.01)))
        if f is F.elu:
            return self.act(x, "elu", float(args[1] if len(args) > 1 else kw.get("alpha", 1.0)))
        if f is F.softplus:
            return self.act(x, "softplus")
        if f in (F.softmax, torch.softmax):
            return self.softmax(x, args[1] if len(args) > 1 else kw.get("dim", -1))
        if f is F.dropout:
            return x
        if f in (torch.flatten, torch.reshape):
            return self.reshape(x, rec)
        if f in (torch.cat, torch.concat):
            return self.cat(list(args[0]), args[1] if len(args) > 1 else kw.get("dim", 0), rec)
        if f in (torch.transpose, torch.permute):
            return self.method("transpose" if f is torch.transpose else "permute", args, kw, rec)
        if f is torch.mean:
            return self.reduce(x, args[1] if len(args) > 1 else kw["dim"], kw.get("keepdim", False), "avg", rec)
        if f is torch.amax:
            return self.reduce(x, args[1] if len(args) > 1 else kw["dim"], kw.get("keepdim", False), "max", rec)
        if f in (F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_max_pool1d, F.adaptive_max_pool2d):
            out = args[1] if len(args) > 1 else kw["output_size"]
            if out not in (1, (1, 1), (1,)):
                raise ValueError("adaptive pool 함수는 출력 크기 1만 지원")
            return self.pool_global(x, "max" if "max" in f.__name__ else "avg", tuple(rec.shape[1:]))
        if f in (torch.unsqueeze, torch.squeeze):
            return self.reshape(x, rec)
        if f is F.interpolate:
            if kw.get("mode", "nearest") != "nearest" or kw.get("scale_factor") is None:
                raise ValueError("interpolate는 nearest, scale_factor만 지원")
            return self.upsample(x, kw["scale_factor"])
        if f is F.pad:
            if kw.get("mode", "constant") != "constant" or kw.get("value", 0) not in (0, None):
                raise ValueError("F.pad는 0으로 채우는 것만 지원")
            return self.pad(x, args[1] if len(args) > 1 else kw["pad"])
        raise ValueError(f"지원하지 않는 함수: {getattr(f, '__name__', f)}")

    def cat(self, xs, dim, rec):
        ts = [t for t in xs if isinstance(t, T)]
        if len(ts) != len(xs):
            raise ValueError("상수 텐서와의 cat은 지원하지 않음")
        nd = len(ts[0].lperm)
        d = dim % (nd + 1) - 1
        if d < 0:
            raise ValueError("배치 축 cat은 지원하지 않음")
        lp = ts[0].lperm
        ts = [self.layout_to(t, lp) for t in ts]
        j = lp[d]
        if j not in (0, nd - 1):
            ts = [self.layout_to(t, _ident(nd)) for t in ts]
            lp, j = _ident(nd), d
            if j not in (0, nd - 1):
                raise ValueError("가운데 축 cat은 지원하지 않음")
        return T(self.g.concat([t.v for t in ts], FIRST if j == 0 and nd > 1 else LAST), lp)

    def method(self, name, args, kw, rec):
        x = args[0]
        if name in IDENT_METHOD:
            return x
        if name in ("view", "reshape", "flatten", "unsqueeze", "squeeze"):
            return self.reshape(x, rec)
        if name in ("permute", "transpose"):
            nd = len(x.lperm)
            if name == "transpose":
                a, b = (d % (nd + 1) for d in args[1:3])
                p = list(range(nd + 1))
                p[a], p[b] = p[b], p[a]
            else:
                p = list(args[1]) if isinstance(args[1], (list, tuple)) else list(args[1:])
                p = [d % (nd + 1) for d in p]
            if p[0] != 0:
                raise ValueError("배치 축을 옮기는 permute/transpose는 지원하지 않음")
            return T(x.v, [x.lperm[d - 1] for d in p[1:]])
        if name in ACT_METHOD:
            return self.act(x, ACT_METHOD[name])
        if name == "softmax":
            return self.softmax(x, args[1] if len(args) > 1 else kw.get("dim", -1))
        if name in BIN_METHOD:
            return self.binary(x, args[1], BIN_METHOD[name], rec)
        if name in ("mean", "amax"):
            return self.reduce(x, args[1] if len(args) > 1 else kw["dim"], kw.get("keepdim", False), "avg" if name == "mean" else "max", rec)
        raise ValueError(f"지원하지 않는 메서드: .{name}()")


def export_graph(model, path, input_shape, quantize=None):
    """input_shape: 표본 하나의 입력 모양 (PyTorch 순서, 배치 제외). 예: 이미지 (3, 32, 32), 시퀀스 (T, F), 토큰 (T,)"""
    ex = Exporter(model, tuple(int(d) for d in input_shape), quantize)
    out = ex.run()
    ex.g.write(path, out)
    return ex.layout
