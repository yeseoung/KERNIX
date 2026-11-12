CC = clang
# -mcpu=apple-m1: int8 sdot(dotprod) 커널 사용 (kernix_model.h). 다른 CPU면 같은 결과의 스칼라 경로로 빌드됨
# src/kernix_runtime.c: 스레드 풀 (kernix_pool_run). 모든 빌드에 같이 넣음
CFLAGS = -O3 -Wall -Wextra -std=c11 -I./include -pthread -lm -mcpu=apple-m1 src/kernix_runtime.c
ACCEL_FLAGS = -DKERNIX_USE_ACCELERATE -framework Accelerate
# kernix.h를 쓰는 모든 소스에 -DKERNIX_USE_METAL을 줘야 함 (float_free가 Metal 할당을 알아야 함)
METAL_FLAGS = $(ACCEL_FLAGS) -DKERNIX_USE_METAL -fobjc-arc src/kernix_metal.m \
              -framework Metal -framework MetalPerformanceShaders -framework Foundation

.PHONY: all lib python-lib clean

all: lib python-lib

# 설치용 정적 라이브러리 (install_mac.sh가 씀). libkernix.a: 스레드 풀 (모든 빌드),
# libkernix_metal.a: Metal 코드 (-DKERNIX_USE_METAL 빌드만, Metal·MPS·Foundation 프레임워크와 같이 링크)
LIB_CFLAGS = -O3 -Wall -Wextra -I./include -mcpu=apple-m1
lib: build/libkernix.a build/libkernix_metal.a

build/libkernix.a: src/kernix_runtime.c include/kernix.h
	@mkdir -p build
	$(CC) $(LIB_CFLAGS) -std=c11 -c src/kernix_runtime.c -o build/kernix_runtime.o
	rm -f $@ && ar rcs $@ build/kernix_runtime.o

build/libkernix_metal.a: src/kernix_metal.m include/kernix.h
	@mkdir -p build
	$(CC) $(LIB_CFLAGS) -DKERNIX_USE_ACCELERATE -DKERNIX_USE_METAL -fobjc-arc -c src/kernix_metal.m -o build/kernix_metal.o
	rm -f $@ && ar rcs $@ build/kernix_metal.o

# Python 바인딩용 공유 라이브러리 (python/kernix.py가 로드). 웹캠 추론 같은 작은 배치는 GPU를 안 쓰므로 Accelerate만
python-lib: src/kernix_capi.c include/kernix.h include/kernix_model.h
	$(CC) $(CFLAGS) $(ACCEL_FLAGS) -dynamiclib src/kernix_capi.c -o python/libkernix.dylib

clean::
	rm -rf build python/libkernix.dylib python/__pycache__

# 테스트·벤치 타깃 (저장소에는 없음, 로컬 test/에만)
-include test/test.mk
