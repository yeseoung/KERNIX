"""MLP 런타임 테스트 데이터 생성과 벤치마크 (make bench-model).

  python test/mlp_bench.py                      테스트/벤치용 모델과 입력·기대 출력을 test/model_data/에 생성
  python test/mlp_bench.py --bench <실행 파일>   KERNIX(런타임, 기존 연산 조합)와 PyTorch CPU/MPS 비교
"""
import os
import resource
import statistics
import struct
import subprocess
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from export_torch import export, floats  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data")
REPEATS = 20
BATCHES = (1, 64, 4096)


def save_tensor(t, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<2I", *t.shape) + floats(t))


def bench_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(784, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10))


def generate():
    os.makedirs(DATA, exist_ok=True)
    torch.manual_seed(0)
    small = nn.Sequential(
        nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128),
        nn.Linear(128, 96, bias=False), nn.SiLU(),
        nn.Linear(96, 96), nn.Tanh(),
        nn.Linear(96, 80), nn.Sigmoid(),
        nn.Linear(80, 32), nn.ReLU(), nn.LayerNorm(32, elementwise_affine=False),
        nn.Linear(32, 10), nn.Softmax(dim=-1),
    )
    nn.init.normal_(small[2].weight, 1.0, 0.5)  # 기본값(1, 0)이면 gamma/beta 버그를 못 잡음
    nn.init.normal_(small[2].bias, 0.0, 0.5)
    wide = nn.Sequential(nn.Linear(4, 9000), nn.SiLU(), nn.GELU(), nn.Linear(9000, 3))
    inplace_first = nn.Sequential(nn.Tanh(), nn.Linear(8, 4))
    cases = (("small", small, 64, (1, 37, 4096)), ("wide", wide, 4, (5,)), ("inplace_first", inplace_first, 8, (3,)))
    with torch.no_grad():
        for name, model, in_dim, batches in cases:
            export(model, os.path.join(DATA, f"{name}.knx"))
            if name == "small":
                export(model, os.path.join(DATA, "small_int8.knx"), quantize="int8")
            for b in batches:
                x = torch.randn(b, in_dim)
                save_tensor(x, os.path.join(DATA, f"{name}_x_{b}.bin"))
                save_tensor(model(x), os.path.join(DATA, f"{name}_y_{b}.bin"))
    export(bench_model(), os.path.join(DATA, "bench.knx"))
    export(bench_model(), os.path.join(DATA, "bench_int8.knx"), quantize="int8")
    print(f"모델과 테스트 데이터 생성: {DATA}")


def torch_measure(model, device, batch):
    """(forward 중앙값 ms, forward당 CPU 시간 ms)"""
    model = model.to(device)
    x = (((torch.arange(batch * 784) % 200) - 100) * 0.01).float().reshape(batch, 784).to(device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    with torch.no_grad():
        model(x); sync()  # 워밍업
        wall = []
        cpu0 = time.process_time()
        for _ in range(REPEATS):
            start = time.perf_counter()
            model(x)
            sync()
            wall.append((time.perf_counter() - start) * 1000)
        cpu = (time.process_time() - cpu0) * 1000 / REPEATS
    return statistics.median(wall), cpu


def bench(binary):
    kernix = {}  # (variant, batch) -> (wall, cpu)
    rss = {}
    for variant in ("runtime", "naive"):  # 프로세스를 따로 띄워 최대 메모리를 분리 측정
        out = subprocess.run([binary, "--bench", variant], check=True, capture_output=True, text=True).stdout
        for line in out.splitlines():
            p = line.split()
            if len(p) == 5 and p[0] == "KERNIX":
                kernix[(p[1], int(p[2]))] = (float(p[3]), float(p[4]))
            elif len(p) == 4 and p[2] == "maxrss_mb":
                rss[p[1]] = float(p[3])

    model = bench_model()
    print(f"\nMLP 784→1024→1024→10 (ReLU), float32, {REPEATS}회 중앙값, PyTorch {torch.__version__}")
    print("naive = 런타임 도입 전 방식(기존 KERNIX 연산을 이어 붙임: matmul → add → relu)\n")
    print("실행 시간 (ms)")
    print(f"{'배치':>6} | {'torch CPU':>10} | {'torch MPS':>10} | {'naive':>9} | {'런타임':>9} | {'vs torch 최선':>12} | {'vs naive':>8}")
    rows = []
    for b in BATCHES:
        cpu_w, cpu_c = torch_measure(model, "cpu", b)
        mps_w, mps_c = torch_measure(model, "mps", b)
        rt_w, rt_c = kernix[("runtime", b)]
        nv_w, nv_c = kernix[("naive", b)]
        rows.append((b, cpu_c, mps_c, nv_c, rt_c))
        print(f"{b:>6} | {cpu_w:>10.3f} | {mps_w:>10.3f} | {nv_w:>9.3f} | {rt_w:>9.3f} | "
              f"{min(cpu_w, mps_w) / rt_w:>11.2f}배 | {nv_w / rt_w:>6.2f}배")
    print("\nforward당 CPU 시간 (ms, 모든 스레드 합계. 적을수록 코어를 덜 씀)")
    print(f"{'배치':>6} | {'torch CPU':>10} | {'torch MPS':>10} | {'naive':>9} | {'런타임':>9}")
    for b, cpu_c, mps_c, nv_c, rt_c in rows:
        print(f"{b:>6} | {cpu_c:>10.3f} | {mps_c:>10.3f} | {nv_c:>9.3f} | {rt_c:>9.3f}")
    py_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)  # macOS는 바이트 단위
    print("\n프로세스 최대 메모리 (MB, 세 배치를 모두 실행한 뒤)")
    print(f"  KERNIX 런타임 {rss['runtime']:.0f} | KERNIX naive {rss['naive']:.0f} | PyTorch(CPU+MPS, 인터프리터 포함) {py_rss:.0f}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--bench":
        bench(sys.argv[2])
    elif len(sys.argv) == 1:
        generate()
    else:
        sys.exit(__doc__)
