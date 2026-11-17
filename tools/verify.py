"""변환한 .knx가 원본 모델과 같은 답을 내는지, 얼마나 빠른지 확인.

  python tools/verify.py model.h5     model.knx [--input x.npy] [--batch 64] [--tol 1e-4] [--ane]
  python tools/verify.py model.tflite model.knx
  python tools/verify.py model.pt     model.knx

원본은 그 프레임워크로 실행한다 (Keras/TFLite는 tensorflow, PyTorch는 torch 필요). 같은 입력을 양쪽에 넣어
최대 상대 오차(max|a-b| / max|a|), 예측(argmax) 일치율, 배치 1·전체 배치 시간을 출력한다.
오차가 --tol을 넘으면 종료 코드 1. int8 .knx나 --ane(float16)는 오차가 원래 크므로 --tol을 직접 올릴 것.
torch.load는 pickle이라 직접 만든 파일만 넣을 것.
"""
import argparse
import os
import statistics
import sys
import time
import unicodedata

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from kernix import Model  # noqa: E402


def load_original(path):
    """원본 모델 → (프레임워크 이름, x(numpy) → 출력(numpy) 함수)"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        import torch
        model = torch.load(path, weights_only=False).eval()

        def run(x):
            with torch.no_grad():
                return model(torch.from_numpy(x)).numpy()
        return "PyTorch", run
    import tensorflow as tf
    if ext in (".h5", ".keras"):
        model = tf.keras.models.load_model(path, compile=False)
        return "Keras", lambda x: np.asarray(model(x, training=False))
    if ext == ".tflite":
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        i, o = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

        # shortcut: 표본 하나씩 float 입력으로 실행 (배치 1 고정 .tflite 기준), 입력까지 양자화된 .tflite면 scale 변환 추가
        def run(x):
            ys = []
            for s in x:
                interp.set_tensor(i, s[None])
                interp.invoke()
                ys.append(interp.get_tensor(o)[0].copy())
            return np.stack(ys)
        return "TFLite", run
    raise ValueError(f"확장자 {ext}: .h5/.keras, .tflite, .pt/.pth만 지원")


def compare(want, got):
    """(최대 상대 오차, 예측 일치율 또는 None). 원본 출력이 전부 0이면 절대 오차"""
    want, got = np.asarray(want, np.float64), np.asarray(got, np.float64).reshape(want.shape)
    scale = np.abs(want).max()
    err = np.abs(want - got).max() / (scale if scale > 0 else 1.0)
    agree = (want.argmax(-1) == got.argmax(-1)).mean() if want.ndim >= 2 and want.shape[-1] >= 2 else None
    return float(err), agree


def ljust(s, w):
    """한글은 터미널에서 두 칸: 표 열을 맞추려고 화면 폭으로 채움"""
    return s + " " * (w - sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s))


def median_ms(fn, x, runs=20):
    """예열 3회 뒤 쉬지 않고 runs회 (쉬면 클럭이 내려가 작은 모델이 느리게 잡힘)"""
    for _ in range(3):
        fn(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - s) * 1000)
    return statistics.median(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description="KERNIX .knx를 원본 모델(Keras/TFLite/PyTorch)과 비교")
    ap.add_argument("original", help=".h5/.keras, .tflite 또는 .pt/.pth")
    ap.add_argument("knx", help="변환한 .knx")
    ap.add_argument("--input", help="입력 .npy (첫 축이 배치). 없으면 표준정규 난수")
    ap.add_argument("--batch", type=int, default=64, help="난수 입력의 표본 수 (기본 64)")
    ap.add_argument("--tol", type=float, default=1e-4, help="허용 최대 상대 오차 (기본 1e-4)")
    ap.add_argument("--ane", action="store_true", help="KERNIX를 ANE 보조로 실행 (float16, 오차 약 1e-3)")
    a = ap.parse_args(argv)
    if a.batch < 1:
        ap.error("--batch는 1 이상")

    try:
        m = Model(a.knx, ane=a.ane)
    except ValueError as e:
        ap.error(str(e))
    if a.input:
        x = np.load(a.input)
        if x.ndim < 2 or tuple(x.shape[1:]) != m.in_shape:
            ap.error(f"--input 모양 {x.shape}: (batch, {', '.join(map(str, m.in_shape))})이어야 함")
        x = np.ascontiguousarray(x, np.float32)
    else:
        x = np.random.default_rng(0).standard_normal((a.batch,) + m.in_shape).astype(np.float32)

    try:
        name, original = load_original(a.original)
    except ImportError as e:
        ap.error(f"원본 실행에 필요한 패키지가 없음 ({e.name}): pip install {'torch' if e.name == 'torch' else 'tensorflow'}")
    except ValueError as e:
        ap.error(str(e))
    try:
        want = original(x)
    except Exception as e:  # noqa: BLE001  원본 실행 실패는 원인과 함께 보고
        hint = "" if a.input else " (Embedding처럼 정수 입력을 받는 모델은 --input으로 실제 입력을 주세요)"
        ap.error(f"원본 실행 실패: {type(e).__name__}: {e}{hint}")
    got = m(x)
    err, agree = compare(want, got)
    n = len(x)

    print(f"입력: {a.input} ({n}개)" if a.input else f"입력: 표준정규 난수 {n}개 (seed 0). 예측 일치율은 실제 데이터(--input)일 때 의미 있음")
    print(f"{'':16}{name:>12}{'KERNIX':>12}")
    print(f"{ljust('최대 상대 오차', 16)}{'':>12}{err:>12.1e}")
    print(f"{ljust('예측 일치율', 16)}{'':>12}{'-' if agree is None else f'{agree * 100:.1f}%':>12}"
          + ("" if agree is None else f"  ({round(agree * n)}/{n})"))
    for b in sorted({1, n}):
        print(f"{ljust(f'배치 {b} 시간', 16)}{median_ms(original, x[:b]):>9.3f} ms{median_ms(m, x[:b]):>9.3f} ms")
    ok = err <= a.tol
    print(f"판정: {'통과' if ok else '실패'} (기준 {a.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
