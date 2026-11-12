""".knx 가중치 블록과 int8 양자화 (docs/int8-quantization-design.md §4, §7). export_torch.py, export_keras.py, knx3.py가 같이 쓴다.

v1(KNX1)은 가중치를 float32 값만 쓰고, v2(KNX2)는 가중치마다 앞에 uint32 dtype을 둔다:
  0 F32    : float32 rows×cols
  1 I8_COL : float32 scale[cols], int8 rows×cols   (w[k][n] ≈ q[k][n]·scale[n], 출력 채널별: KERNIX가 int8로 계산)
  2 I8_ROW : float32 scale[rows], int8 rows×cols   (w[k][n] ≈ q[k][n]·scale[k], 입력 특징별: 로드할 때 float로 풂)
"""
import numpy as np

MAGIC = {None: b"KNX1", "int8": b"KNX2"}
DTYPE = {"f32": 0, "i8_col": 1, "i8_row": 2}


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def quantize(w, axis):
    """대칭 int8: axis "col"이면 열마다, "row"면 행마다 scale = max|w| / 127 (모두 0이면 1). → (scale float32, q int8)"""
    w = np.asarray(w, np.float64)
    m = np.abs(w).max(axis=0 if axis == "col" else 1)
    scale = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    s = scale[None, :] if axis == "col" else scale[:, None]
    q = np.clip(np.rint(w / s), -127, 127).astype(np.int8)
    return scale, q


def weight_block(w, mode, v2=True):
    """가중치 w(rows×cols, in×out 방향)의 바이트. v2가 아니면 dtype 없이 float32 값만 (KNX1)"""
    if not v2:
        if mode != "f32":
            raise ValueError("KNX1은 float32 가중치만 가능")
        return f32(w)
    head = np.uint32(DTYPE[mode]).astype("<u4").tobytes()
    if mode == "f32":
        return head + f32(w)
    scale, q = quantize(w, "col" if mode == "i8_col" else "row")
    return head + f32(scale) + np.ascontiguousarray(q).tobytes()
