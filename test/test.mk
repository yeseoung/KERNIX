# 공개 테스트 (CI: .github/workflows/test.yml). Makefile이 include함. `make test`로 전부 실행
#   PY: torch가 있는 파이썬 (기본 .venv/bin/python), KERAS_PY: tensorflow가 있는 파이썬 (기본 PY)
PY ?= .venv/bin/python
KERAS_PY ?= $(PY)

.PHONY: test test-pool test-model test-zoo test-zoo-keras test-fuzz test-verify

test: test-pool test-model test-zoo test-zoo-keras test-fuzz test-verify

# 스레드 풀 검사 (assert는 -DNDEBUG 없이 빌드해야 동작)
test-pool: test/pool_test.c src/kernix_runtime.c include/kernix.h
	$(CC) $(CFLAGS) test/pool_test.c -o test/POOL_test
	./test/POOL_test

# 모델 런타임 (C): 검사 모델·데이터를 test/model_data/에 만들고 Metal 빌드와 Accelerate 전용 빌드 모두 검증
test-model: test/model_test.c test/mlp_bench.py src/kernix_metal.m include/kernix.h include/kernix_model.h
	$(PY) test/mlp_bench.py
	$(CC) $(CFLAGS) $(METAL_FLAGS) test/model_test.c -o test/MODEL_metal
	./test/MODEL_metal --check
	$(CC) $(CFLAGS) $(ACCEL_FLAGS) test/model_test.c -o test/MODEL_accel
	./test/MODEL_accel --check

# 모델 모음: 레이어 종류마다 작은 모델을 내보내 KERNIX 출력이 PyTorch/Keras와 같은지 (KNX3 그래프 포함)
test-zoo: python-lib
	$(PY) test/zoo_torch_test.py

test-zoo-keras: python-lib
	$(KERAS_PY) test/zoo_keras_test.py

# .knx 로더 퍼징: 손상 파일 수천 개를 ASan/UBSan 빌드에 넣음
test-fuzz: test/knx_fuzz.c include/kernix.h include/kernix_model.h
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -std=c11 -I./include -pthread -lm \
	      -mcpu=apple-m1 src/kernix_runtime.c $(ACCEL_FLAGS) test/knx_fuzz.c -o test/FUZZ_bin
	$(PY) test/knx_fuzz_test.py ./test/FUZZ_bin

# tools/verify.py: 원본 대비 검증 도구 (작은 PyTorch 모델로 통과·실패·입력 파일·모양 오류)
test-verify: python-lib
	$(PY) test/verify_test.py

clean::
	rm -rf test/model_data test/FUZZ_bin.dSYM
	rm -f test/MODEL_metal test/MODEL_accel test/POOL_test test/FUZZ_bin
