"""tools/verify.py 검사: 작은 PyTorch 모델로 통과/실패/입력 파일/모양 오류/회귀 출력.
  make test-verify   (먼저 make python-lib)
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from export_torch import convert  # noqa: E402
from verify import main  # noqa: E402

torch.manual_seed(0)
D = tempfile.mkdtemp()


def make(name, model, input_shape):
    model.eval()
    pt, knx = os.path.join(D, name + ".pt"), os.path.join(D, name + ".knx")
    torch.save(model, pt)
    with contextlib.redirect_stdout(io.StringIO()):
        convert(pt, knx, None, input_shape)
    return pt, knx


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(argv)
    return code, out.getvalue()


cls = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 6 * 6, 5))
CLS_PT, CLS_KNX = make("cls", cls, (3, 8, 8))
REG_PT, REG_KNX = make("reg", nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1)), None)


def test_pass():
    code, out = run([CLS_PT, CLS_KNX])
    assert code == 0 and "통과" in out, out


def test_fail_on_tampered():
    bad = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 6 * 6, 5))
    bad.load_state_dict(cls.state_dict())
    with torch.no_grad():
        bad[0].weight[0, 0, 0, 0] += 1
    pt = os.path.join(D, "bad.pt")
    torch.save(bad.eval(), pt)
    code, out = run([pt, CLS_KNX])
    assert code == 1 and "실패" in out, out


def test_input_file():
    npy = os.path.join(D, "x.npy")
    np.save(npy, np.random.default_rng(1).normal(size=(7, 3, 8, 8)).astype(np.float32))
    code, out = run([CLS_PT, CLS_KNX, "--input", npy])
    assert code == 0 and "7" in out, out


def test_bad_input_shape():
    npy = os.path.join(D, "bad.npy")
    np.save(npy, np.zeros((4, 3, 9, 9), np.float32))
    try:
        run([CLS_PT, CLS_KNX, "--input", npy])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("모양 오류인데 SystemExit 2가 아님")


def test_regression_no_agreement():
    code, out = run([REG_PT, REG_KNX])
    line = next(ln for ln in out.splitlines() if "예측 일치율" in ln)
    assert code == 0 and "-" in line, out


fail = 0
for name, fn in list(globals().items()):
    if name.startswith("test_"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                fn()
            print(f"  ✓ {name}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
print("통과 ✓" if not fail else f"실패 {fail}개")
sys.exit(1 if fail else 0)
