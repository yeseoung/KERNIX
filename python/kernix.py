"""KERNIX 모델 런타임 Python 바인딩 (ctypes, numpy만 필요). 먼저 `make python-lib`으로 libkernix.dylib를 만든다.

    from kernix import Model
    m = Model("model.knx")
    y = m(x)   # x: (batch, *m.in_shape) float32 → (batch, *m.out_shape)
               # 예: MLP (batch, in), 시퀀스 (batch, T, F), 이미지 (batch, H, W, C) 또는 PyTorch 모델이면 (batch, C, H, W)
               # 이전 방식 (rows, in_dim)도 받음 (rows는 표본당 행 수 seq_len의 배수)

한 Model을 여러 스레드에서 불러도 된다: 호출은 모델마다 하나씩 차례로 실행된다 (내부 버퍼를 재사용하므로).
스레드마다 동시에 돌리려면 스레드별로 Model을 따로 만든다.
"""
import ctypes
import os
import threading
import warnings

import numpy as np

_LIB_PATH = os.environ.get("KERNIX_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkernix.dylib"))
_lib = ctypes.CDLL(_LIB_PATH)
_lib.kernix_capi_load.argtypes = [ctypes.c_char_p]
_lib.kernix_capi_load.restype = ctypes.c_void_p
_lib.kernix_capi_free.argtypes = [ctypes.c_void_p]
_lib.kernix_capi_free.restype = None
_lib.kernix_capi_dims.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 3
_lib.kernix_capi_dims.restype = None
_lib.kernix_capi_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
_lib.kernix_capi_forward.restype = ctypes.c_int
_lib.kernix_capi_trim.argtypes = [ctypes.c_void_p]
_lib.kernix_capi_trim.restype = None
_lib.kernix_capi_set_workspace_limit.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.kernix_capi_set_workspace_limit.restype = None
_lib.kernix_capi_shapes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 5
_lib.kernix_capi_shapes.restype = None
_lib.kernix_capi_set_ane.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.kernix_capi_set_ane.restype = ctypes.c_int
_lib.kernix_capi_ane_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.kernix_capi_ane_info.restype = None
_lib.kernix_capi_ane_dump.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
_lib.kernix_capi_ane_dump.restype = ctypes.c_int


class Model:
    def __init__(self, path, workspace_mb=None, ane=False):
        """workspace_mb: 작업 버퍼 상한(MB). 이보다 큰 배치는 나눠 실행해 최대 메모리를 묶는다.
        None이면 기본값(64 MB), 0이면 제한 없음(배치 전체를 한 번에)
        ane: True면 합성곱 구간을 Core ML로 뉴럴엔진에서 실행 (float16이라 오차 약 1e-3, 기본은 꺼짐).
        컴파일은 뒤에서 하고 그동안은 KERNIX 경로로 답하며, 구간마다 실제 호출 시간으로 빠른 쪽을 고른다"""
        # ctypes는 C 호출 동안 GIL을 풀어 주므로, 잠금이 없으면 다른 스레드가 같은 작업 버퍼를 덮어써 결과가 조용히 틀림.
        # 맨 먼저 만든다: 아래에서 예외가 나도 close()가 잠금을 쓸 수 있게
        self._lock = threading.Lock()
        self._h = _lib.kernix_capi_load(os.fsencode(path))
        if not self._h:
            raise ValueError(f"KERNIX 모델을 읽지 못함: {path}")
        if workspace_mb is not None:
            if workspace_mb < 0:
                raise ValueError("workspace_mb는 0 이상")
            _lib.kernix_capi_set_workspace_limit(self._h, int(workspace_mb * (1 << 20)))
        if ane and _lib.kernix_capi_set_ane(self._h, 1) != 0:
            warnings.warn("ANE 보조를 쓸 수 없음 (macOS 13 이상, Core ML 빌드 필요): KERNIX 경로로 실행", RuntimeWarning, stacklevel=2)
        dims = [ctypes.c_int() for _ in range(3)]
        _lib.kernix_capi_dims(self._h, *dims)
        self.in_dim, self.out_dim, self.seq_len = (d.value for d in dims)
        in_nd, out_nd, layout = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        in_s, out_s = (ctypes.c_int * 4)(), (ctypes.c_int * 4)()
        _lib.kernix_capi_shapes(self._h, ctypes.byref(in_nd), in_s, ctypes.byref(out_nd), out_s, ctypes.byref(layout))
        self._in_cl, self._out_cl = tuple(in_s[:in_nd.value]), tuple(out_s[:out_nd.value])  # channels-last (KERNIX 내부)
        # PyTorch 합성곱 모델(layout 1)은 채널이 앞 (C, H, W): 바인딩이 채널 축을 옮겨 PyTorch와 같은 모양으로 주고받음
        self.channels_first = layout.value == 1
        cf = lambda s: s[-1:] + s[:-1] if self.channels_first and len(s) >= 2 else s  # noqa: E731
        self.in_shape, self.out_shape = cf(self._in_cl), cf(self._out_cl)
        self._forward = _lib.kernix_capi_forward  # 속성 조회를 줄여 호출당 오버헤드 감소

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim >= 2 and tuple(x.shape[1:]) == self.in_shape:
            if self.channels_first and x.ndim >= 3:
                x = np.moveaxis(x, 1, -1)  # (B, C, ...) → (B, ..., C)
        elif not (x.ndim == 2 and x.shape[1] == self.in_dim):  # 이전 방식: (rows, in_dim)
            raise ValueError(f"입력 모양 {x.shape}: (batch, {', '.join(map(str, self.in_shape))}) 또는 (rows, {self.in_dim})이어야 함")
        x = np.ascontiguousarray(x, dtype=np.float32)  # 이미 float32 연속 배열이면 복사 없음
        rows = x.size // self.in_dim
        if rows == 0 or rows % self.seq_len:
            raise ValueError(f"입력 행 수 {rows}는 seq_len {self.seq_len}의 양의 배수여야 함")
        out = np.empty((rows // self.seq_len,) + self._out_cl, np.float32)
        with self._lock:
            status = self._forward(self._h, x.ctypes.data, rows, out.ctypes.data)
        if status != 0:
            raise RuntimeError("KERNIX forward 실패")
        return np.moveaxis(out, -1, 1) if self.channels_first and out.ndim >= 3 else out

    def ane_info(self):
        """ANE 보조 상태: segments(구간 수), compiles(시작한 컴파일 수), ane_runs(ANE로 실행한 구간 수), failures"""
        out = (ctypes.c_int * 4)()
        with self._lock:
            _lib.kernix_capi_ane_info(self._h, out)
        return dict(zip(("segments", "compiles", "ane_runs", "failures"), out))

    def _ane_dump(self, seg, batch, path):
        """테스트용: 구간 seg를 표본 batch개 고정의 Core ML 모델(.mlmodel)로 path에 씀. 성공 0, 실패 -1"""
        with self._lock:
            return _lib.kernix_capi_ane_dump(self._h, seg, batch, os.fsencode(path))

    def trim(self):
        """작업 버퍼를 운영체제에 돌려준다 (가중치는 유지). 큰 배치 뒤 쉬는 동안 부르면 좋고, 다음 호출이 다시 할당"""
        with self._lock:
            _lib.kernix_capi_trim(self._h)

    def close(self):
        if getattr(self, "_h", None):  # __init__이 중간에 실패한 객체도 정리
            with self._lock:
                _lib.kernix_capi_free(self._h)
                self._h = None

    __del__ = close
