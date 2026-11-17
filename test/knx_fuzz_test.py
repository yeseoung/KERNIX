"""KNX3 로더 퍼징: 정상 모델(연산 종류마다)을 잘라 내거나 바이트를 바꾼 파일을 ASan/UBSan 하네스(test/knx_fuzz.c)에 넣는다.
  python test/knx_fuzz_test.py <하네스 실행 파일>     (make test-fuzz)
손상 파일은 거부되거나, 받아들여지면 forward가 메모리 오류 없이 끝나야 한다. 정상 파일은 모두 로드돼야 한다.
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from knx3 import FIRST, LAST, Graph  # noqa: E402

rng = np.random.default_rng(0)
W = lambda *s: rng.normal(0, 0.3, s)  # noqa: E731


def models():
    g = Graph((9, 8, 3))  # CNN: 합성곱 3종, 풀링, 패딩, 업샘플, 잔차, 전역 풀링
    x = g.conv(0, W(27, 6), W(6), (3, 3), pads=(1, 1, 1, 1), act="relu")
    y = g.conv(x, W(9, 6), None, (3, 3), pads=(1, 1, 1, 1), groups=6)
    y = g.conv(y, W(27, 6), W(6), (3, 3), pads=(1, 1, 1, 1), groups=2)
    x = g.binary(x, y, "add")
    x = g.upsample(g.pad(g.pool(x, "max", (2, 2), (2, 2)), (1, 0, 0, 1)), 2, 2)
    x = g.pool(x, "avg", (3, 3), (1, 1), (1, 1, 1, 1), True)
    yield "cnn", g, g.linear(g.global_pool(x, "avg"), W(6, 4), W(4), act="softmax")
    g = Graph((7, 5))  # RNN 3종, 양방향, 고르기, 이어 붙이기
    a = g.rnn(0, "GRU", 4, [(W(5, 12), W(4, 12), W(24)), (W(5, 12), W(4, 12), W(24))], ret_seq=True)
    b = g.rnn(0, "LSTM", 3, [(W(5, 12), W(3, 12), W(12))], act="relu", ret_seq=True)
    c = g.concat([a, b], LAST)
    c = g.rnn(c, "RNN", 4, [(W(11, 4), W(4, 4), W(4)), (W(11, 4), W(4, 4), W(4))], ret_seq=False, merge="ave")
    yield "rnn", g, g.concat([c, g.select(a, FIRST, 6, 1, True)], LAST)
    g = Graph((6,))  # 임베딩 + 어텐션 + 모양 바꾸기
    e = g.add_const(g.embedding(0, W(10, 8)), W(48))
    h = g.layernorm(g.binary(e, g.attention(e, 2, W(8, 6), W(6), W(8, 6), W(6), W(8, 4), W(4), W(4, 8), W(8), causal=True), "add"),
                    W(8), W(8), 1e-5)
    h = g.permute(g.reshape(h, (6, 2, 4)), (1, 0, 2))
    yield "attn", g, g.affine(g.reshape(h, (2, 24)), W(24), W(24))


def mutations(data, n):
    words = len(data) // 4
    for i in range(n):
        b = bytearray(data)
        kind = i % 4
        if kind == 0:  # 잘라 내기
            yield bytes(b[: rng.integers(0, len(b))])
        elif kind == 1:  # u32 하나를 경계값으로
            w = rng.integers(0, min(words, 400))
            b[4 * w:4 * w + 4] = struct.pack("<I", int(rng.choice([0, 1, 2, 3, 7, 255, 65535, 65536, 2**31 - 1, 2**31, 2**32 - 1])))
            yield bytes(b)
        elif kind == 2:  # 앞부분 바이트 무작위
            for _ in range(rng.integers(1, 4)):
                b[rng.integers(0, min(len(b), 1600))] = rng.integers(0, 256)
            yield bytes(b)
        else:  # 뒤에 쓰레기
            yield bytes(b) + bytes(rng.integers(0, 256, rng.integers(1, 9)).astype(np.uint8))


def main(harness):
    with tempfile.TemporaryDirectory() as d:
        good, bad = [], []
        for name, g, out in models():
            p = os.path.join(d, f"{name}.knx")
            g.write(p, out)
            good.append(p)
            data = open(p, "rb").read()
            for i, m in enumerate(mutations(data, 1500)):
                q = os.path.join(d, f"{name}_{i}.knx")
                open(q, "wb").write(m)
                bad.append(q)
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
        r = subprocess.run([harness] + good, capture_output=True, text=True, env=env)
        ok_good = r.returncode == 0 and f"loaded {len(good)} / {len(good)}" in r.stdout
        print(f"  {'✓' if ok_good else '✗'} 정상 모델 {len(good)}개 로드·실행 ({r.stdout.strip()})")
        fails, accepted = 0, 0
        for i in range(0, len(bad), 500):  # 한 번에 500개씩
            r = subprocess.run([harness] + bad[i:i + 500], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                fails += 1
                print("  ✗ 손상 파일에서 오류:\n" + "\n".join(r.stderr.splitlines()[-25:]))
                break
            accepted += int(r.stdout.split()[1])
        print(f"  {'✓' if not fails else '✗'} 손상 파일 {len(bad)}개: 메모리 오류 없음 "
              f"(거부 {len(bad) - accepted}개, 값만 바뀌어 로드·실행까지 간 것 {accepted}개)")
        sys.exit(0 if ok_good and not fails else 1)


if __name__ == "__main__":
    main(sys.argv[1])
