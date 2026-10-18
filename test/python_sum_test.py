
import time
import statistics
import torch

N = 100_000_000
REPEATS = 20
THREADS = 8

# CPU 연산 스레드 수 설정
torch.set_num_threads(THREADS)

# 텐서 생성 및 초기화 (측정 시간에서 제외)
tensor = torch.ones(N, dtype=torch.float32)

print(f"원소 개수: {tensor.numel():,}")
print(f"자료형: {tensor.dtype}")
print(f"장치: {tensor.device}")
print(f"스레드 수: {torch.get_num_threads()}")

# 워밍업: 측정 시간에서 제외
result = torch.sum(tensor).item()

if result != float(N):
    raise RuntimeError(f"합계 오류: {result}")

times = []

for run in range(REPEATS):
    start = time.perf_counter()

    result = torch.sum(tensor).item()

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    print(
        f"{run + 1}회차: "
        f"합계={result:.1f}, "
        f"시간={elapsed:.6f}초"
    )

print(f"\n평균 시간: {statistics.mean(times):.6f}초")
print(f"중앙값: {statistics.median(times):.6f}초")
