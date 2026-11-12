"""KNX3 그래프 작성기 (docs/graph-runtime-design.md). keras_graph.py와 torch_graph.py가 같이 쓴다.

값 번호: 0 = 모델 입력, 연산 k의 출력 = k+1. 모양은 표본 하나(배치 제외)의 channels-last 모양이다
(이미지 H×W×C, 시퀀스 T×F, 벡터 F). C 로더(include/kernix_model.h kernix_infer)와 같은 규칙으로 모양을 추적한다.
"""
import struct

import numpy as np

from knx_quant import f32, weight_block

OP = dict(LINEAR=0, ACT=1, LAYERNORM=2, SOFTMAX=3, LSTM=4, GRU=5, RNN=6, CONV=7, POOL=8, GLOBAL_POOL=9, RESHAPE=10,
          PERMUTE=11, BINARY=12, CONCAT=13, EMBEDDING=14, AFFINE=15, ADD_CONST=16, ATTENTION=17, SELECT=18, PAD=19,
          UPSAMPLE=20)
ACT = dict(relu=0, gelu=1, silu=2, sigmoid=3, tanh=4, leaky_relu=5, elu=6, selu=7, softplus=8, relu6=9,
           hard_sigmoid=10, hard_swish=11, gelu_tanh=12, exp=13, mish=14, softsign=15)
BIN = dict(add=0, sub=1, mul=2, div=3, max=4, min=5)
MERGE = dict(concat=0, sum=1, mul=2, ave=3)
FIRST, LAST = 0, 1  # CONCAT/SELECT 축: 표본의 첫 축, 마지막(채널) 축
GATES = {"LSTM": 4, "GRU": 3, "RNN": 1}


class Graph:
    def __init__(self, in_shape, layout=0, quantize=None):
        """in_shape: 표본 하나의 입력 모양 (channels-last). layout 1 = PyTorch channels-first 입력 (바인딩이 축을 옮김)"""
        if quantize not in (None, "int8"):
            raise ValueError(f"지원하지 않는 quantize: {quantize}")
        self.layout, self.quantize = layout, quantize
        self.shapes = [tuple(int(d) for d in in_shape)]
        self.ops = []  # dict(kind, inp, u32, blobs, act)

    # ---- 내부 ----
    def shape(self, v):
        return self.shapes[v]

    def _add(self, kind, inp, u32, blobs, shape):
        shape = tuple(int(d) for d in shape)
        if not 1 <= len(shape) <= 4 or min(shape) < 1:
            raise ValueError(f"{kind}: 출력 모양 {shape}을 KERNIX에서 표현할 수 없음 (축 1~4개, 크기 ≥ 1)")
        self.ops.append(dict(kind=kind, inp=list(inp), u32=[int(x) for x in u32], blobs=blobs))
        self.shapes.append(shape)
        return len(self.ops)

    def _mode(self, mode):
        return mode if self.quantize else "f32"

    # ---- 연산 ----
    def act(self, x, name, alpha=0.0):
        if name in (None, "linear"):
            return x
        if name == "softmax":
            return self.softmax(x)
        return self._add("ACT", [x], [ACT[name]], [("fl", alpha)], self.shape(x))

    def linear(self, x, w, b=None, act=None, alpha=0.0):
        """y = x·w + b (마지막 축). w: in×out"""
        w = np.asarray(w, np.float64)
        s = self.shape(x)
        if s[-1] != w.shape[0]:
            raise ValueError(f"Linear 입력 {s[-1]} != 가중치 {w.shape}")
        blobs = [("w", w, "i8_col")] + ([("f", np.asarray(b, np.float64))] if b is not None else [])
        y = self._add("LINEAR", [x], [w.shape[0], w.shape[1], b is not None], blobs, s[:-1] + (w.shape[1],))
        return self.act(y, act, alpha)

    def layernorm(self, x, gamma, beta, eps):
        c = self.shape(x)[-1]
        g = np.ones(c) if gamma is None else gamma
        bt = np.zeros(c) if beta is None else beta
        return self._add("LAYERNORM", [x], [c, 1], [("fl", eps), ("f", g), ("f", bt)], self.shape(x))

    def softmax(self, x):
        return self._add("SOFTMAX", [x], [], [], self.shape(x))

    def rnn(self, x, cell, units, dirs_w, act="tanh", ret_seq=False, merge="concat"):
        """cell: LSTM/GRU/RNN. dirs_w: 방향마다 (W in×G·u, U u×G·u, bias). 게이트 순서는 Keras (LSTM i f c o, GRU z r h),
        GRU bias는 입력 쪽 3u + 순환 쪽 3u (reset_after)"""
        s = self.shape(x)
        if len(s) != 2:
            raise ValueError(f"{cell} 입력은 (T, F)여야 함: {s}")
        blobs = []
        for w, u, b in dirs_w:
            blobs += [("w", w, "i8_row"), ("w", u, "i8_col"), ("f", b)]
        uo = units * (2 if len(dirs_w) == 2 and merge == "concat" else 1)
        return self._add(cell, [x], [s[1], units, ACT[act], int(ret_seq), len(dirs_w), MERGE[merge]], blobs,
                         (s[0], uo) if ret_seq else (uo,))

    @staticmethod
    def _hw(s):
        if len(s) not in (2, 3):
            raise ValueError(f"합성곱·풀링 입력은 (T, C) 또는 (H, W, C)여야 함: {s}")
        return s[0], (s[1] if len(s) == 3 else 1)

    def conv(self, x, w, b, k, stride=(1, 1), dil=(1, 1), pads=(0, 0, 0, 0), groups=1, act=None, alpha=0.0):
        """w: (kh·kw·cin/groups) × cout, 행 순서 (ky, kx, ci). pads: (위, 아래, 왼, 오른)"""
        s = self.shape(x)
        H, W = self._hw(s)
        kh, kw = k
        cout = w.shape[1]
        oh = (H + pads[0] + pads[1] - dil[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (W + pads[2] + pads[3] - dil[1] * (kw - 1) - 1) // stride[1] + 1
        out = (oh, ow, cout) if len(s) == 3 else (oh, cout)
        blobs = [("w", np.asarray(w, np.float64), "i8_col")] + ([("f", np.asarray(b, np.float64))] if b is not None else [])
        y = self._add("CONV", [x], [s[-1], cout, kh, kw, *stride, *dil, *pads, groups, b is not None], blobs, out)
        return self.act(y, act, alpha)

    def pool(self, x, kind, k, stride, pads=(0, 0, 0, 0), count_pad=False):
        s = self.shape(x)
        H, W = self._hw(s)
        oh = (H + pads[0] + pads[1] - k[0]) // stride[0] + 1
        ow = (W + pads[2] + pads[3] - k[1]) // stride[1] + 1
        out = (oh, ow, s[-1]) if len(s) == 3 else (oh, s[-1])
        return self._add("POOL", [x], [{"max": 0, "avg": 1}[kind], *k, *stride, *pads, int(count_pad)], [], out)

    def global_pool(self, x, kind):
        return self._add("GLOBAL_POOL", [x], [{"max": 0, "avg": 1}[kind]], [], (self.shape(x)[-1],))

    def reshape(self, x, shape):
        shape = tuple(int(d) for d in shape)
        if shape == self.shape(x):
            return x
        if int(np.prod(shape)) != int(np.prod(self.shape(x))):
            raise ValueError(f"Reshape {self.shape(x)} → {shape}: 크기가 다름")
        return self._add("RESHAPE", [x], [len(shape), *shape], [], shape)

    def permute(self, x, perm):
        perm = tuple(perm)
        if perm == tuple(range(len(perm))):
            return x
        s = self.shape(x)
        return self._add("PERMUTE", [x], [len(perm), *perm], [], tuple(s[p] for p in perm))

    def binary(self, a, b, op):
        sa, sb = self.shape(a), self.shape(b)
        ra, rb = int(np.prod(sa[:-1])), int(np.prod(sb[:-1]))
        big = sa if ra * sa[-1] >= rb * sb[-1] else sb
        R, C = max(ra, rb), max(sa[-1], sb[-1])
        if ra not in (R, 1) or rb not in (R, 1) or sa[-1] not in (C, 1) or sb[-1] not in (C, 1) or int(np.prod(big[:-1])) != R:
            raise ValueError(f"이항 연산 모양 {sa}, {sb}: KERNIX 브로드캐스트(행 또는 채널 방향 크기 1)로 맞출 수 없음")
        return self._add("BINARY", [a, b], [BIN[op]], [], big[:-1] + (C,))

    def concat(self, xs, axis):
        if len(xs) == 1:
            return xs[0]
        if len(xs) > 8:  # 입력 8개씩 나눠 이어 붙임
            return self.concat([self.concat(xs[i:i + 8], axis) for i in range(0, len(xs), 8)], axis)
        s0 = self.shape(xs[0])
        ax = 0 if axis == FIRST else len(s0) - 1
        out = list(s0)
        out[ax] = sum(self.shape(v)[ax] for v in xs)
        return self._add("CONCAT", xs, [axis], [], out)

    def embedding(self, x, table):
        table = np.asarray(table, np.float64)
        return self._add("EMBEDDING", [x], [table.shape[0], table.shape[1]], [("w", table, "i8_row")],
                         self.shape(x) + (table.shape[1],))

    def affine(self, x, scale, shift):
        """y = x·scale + shift (채널마다)"""
        c = self.shape(x)[-1]
        scale = np.broadcast_to(np.asarray(scale, np.float64), (c,))
        shift = np.broadcast_to(np.asarray(shift, np.float64), (c,))
        return self._add("AFFINE", [x], [c], [("f", scale), ("f", shift)], self.shape(x))

    def add_const(self, x, c):
        c = np.asarray(c, np.float64).reshape(-1)
        if c.size != int(np.prod(self.shape(x))):
            raise ValueError(f"상수 크기 {c.size} != 입력 {self.shape(x)}")
        return self._add("ADD_CONST", [x], [c.size], [("f", c)], self.shape(x))

    def attention(self, q, heads, wq, bq, wk, bk, wv, bv, wo, bo, v=None, k=None, causal=False):
        """w*: 입력 채널 × heads·d (헤드 순서대로), wo: heads·dv × out"""
        inp = [q] + ([v] if v is not None else []) + ([k] if k is not None else [])
        dk, dv = wq.shape[1] // heads, wv.shape[1] // heads
        cv, ck = self.shape(v if v is not None else q)[-1], self.shape(k if k is not None else (v if v is not None else q))[-1]
        has_b = bq is not None
        blobs = []
        for w, b in ((wq, bq), (wk, bk), (wv, bv), (wo, bo)):
            blobs.append(("w", np.asarray(w, np.float64), "i8_col"))
            if has_b:
                blobs.append(("f", np.asarray(b, np.float64)))
        return self._add("ATTENTION", inp, [heads, dk, dv, int(causal), wq.shape[0], cv, ck, wo.shape[1], int(has_b)], blobs,
                         self.shape(q)[:-1] + (wo.shape[1],))

    def select(self, x, axis, start, length, squeeze=False):
        s = list(self.shape(x))
        ax = 0 if axis == FIRST else len(s) - 1
        if start < 0 or start + length > s[ax]:
            raise ValueError(f"Select 범위 [{start}, {start + length}) 밖: {s}")
        if start == 0 and length == s[ax] and not squeeze:
            return x
        s[ax] = length
        if squeeze and length == 1 and len(s) > 1:
            del s[ax]
        return self._add("SELECT", [x], [axis, start, length, int(squeeze)], [], s)

    def pad(self, x, pads):
        s = self.shape(x)
        if not any(pads):
            return x
        out = (s[0] + pads[0] + pads[1],) + ((s[1] + pads[2] + pads[3],) if len(s) == 3 else ()) + (s[-1],)
        return self._add("PAD", [x], list(pads), [], out)

    def upsample(self, x, sh, sw=1):
        s = self.shape(x)
        out = (s[0] * sh,) + ((s[1] * sw,) if len(s) == 3 else ()) + (s[-1],)
        return self._add("UPSAMPLE", [x], [sh, sw], [], out)

    # ---- 저장 ----
    def _fold(self, out_v):
        """LINEAR/CONV 바로 뒤의 AFFINE(BatchNorm 등)을 가중치에 접어 넣음: 그 출력을 AFFINE만 읽을 때"""
        cons = [0] * (len(self.ops) + 1)
        for op in self.ops:
            for i in op["inp"]:
                cons[i] += 1
        cons[out_v] += 1
        alias = list(range(len(self.ops) + 1))
        keep = []
        for k, op in enumerate(self.ops):
            op["inp"] = [alias[i] for i in op["inp"]]
            src = op["inp"][0]
            if op["kind"] == "AFFINE" and src > 0 and cons[src] == 1 and self.ops[src - 1]["kind"] in ("LINEAR", "CONV"):
                p = self.ops[src - 1]
                scale, shift = op["blobs"][0][1], op["blobs"][1][1]
                w = p["blobs"][0][1] * scale[None, :]
                b = (p["blobs"][1][1] if len(p["blobs"]) > 1 else np.zeros(w.shape[1])) * scale + shift
                p["blobs"] = [("w", w, p["blobs"][0][2]), ("f", b)]
                p["u32"][2 if p["kind"] == "LINEAR" else 13] = 1
                alias[k + 1] = src
                cons[src] = cons[k + 1]  # 접힌 값의 소비자가 이제 src를 읽음 (다음 접기 판단에 필요)
                continue
            keep.append(k)
        # 번호 다시 매기기
        new = {0: 0}
        ops = []
        for k in keep:
            op = self.ops[k]
            op["inp"] = [new[alias[i]] for i in op["inp"]]
            ops.append(op)
            new[k + 1] = len(ops)
        return ops, new[alias[out_v]]

    def write(self, path, out=None):
        """out: 출력 값 번호 (기본: 마지막 연산). 출력이 마지막 연산이 아니면 그 뒤 연산은 버림"""
        out = len(self.ops) if out is None else out
        ops, out = self._fold(out)
        live, stack = set(), [out]  # 출력에 쓰이는 연산만 남김
        while stack:
            v = stack.pop()
            if v > 0 and v not in live:
                live.add(v)
                stack += ops[v - 1]["inp"]
        if not live:
            raise ValueError("연산이 없는 모델")
        new = {0: 0}
        kept = []
        for v in sorted(live):
            op = ops[v - 1]
            op["inp"] = [new[i] for i in op["inp"]]
            kept.append(op)
            new[v] = len(kept)
        ops = kept
        body = []
        for op in ops:
            b = struct.pack("<2I", OP[op["kind"]], len(op["inp"])) + struct.pack(f"<{len(op['inp'])}I", *op["inp"])
            b += struct.pack(f"<{len(op['u32'])}I", *op["u32"])
            for blob in op["blobs"]:
                if blob[0] == "w":
                    b += weight_block(blob[1], self._mode(blob[2]), True)
                elif blob[0] == "fl":
                    b += struct.pack("<f", blob[1])
                else:
                    b += f32(blob[1])
            body.append(b)
        s = self.shapes[0]
        with open(path, "wb") as f:
            f.write(b"KNX3" + struct.pack("<4I", len(body), self.layout, len(s), *s[:1]) + struct.pack(f"<{len(s) - 1}I", *s[1:]) + b"".join(body))
