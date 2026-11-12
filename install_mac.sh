#!/usr/bin/env bash
# KERNIX macOS 설치: 헤더(kernix.h, kernix_model.h)와 정적 라이브러리(libkernix.a, libkernix_metal.a)를 PREFIX에 설치.
#   ./install_mac.sh                  /usr/local에 설치 (쓰기 권한이 없으면 sudo로)
#   PREFIX=~/.local ./install_mac.sh  다른 위치에 설치
#   ./install_mac.sh --uninstall      설치한 파일 삭제
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-/usr/local}"
HEADERS=(kernix.h kernix_model.h)
LIBS=(libkernix.a libkernix_metal.a)

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "오류: macOS 전용입니다." >&2
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "경고: Apple Silicon(arm64)이 아닙니다. 빌드는 -mcpu=apple-m1을 쓰므로 실패하거나 다른 CPU에서 돌지 않을 수 있습니다." >&2
fi

# PREFIX에 쓸 수 없으면 sudo (가장 가까운 존재하는 상위 폴더로 판단)
SUDO=""
dir="$PREFIX"
while [[ ! -e "$dir" ]]; do dir="$(dirname "$dir")"; done
[[ -w "$dir" ]] || SUDO="sudo"

if [[ "${1:-}" == "--uninstall" ]]; then
    for h in "${HEADERS[@]}"; do $SUDO rm -f "$PREFIX/include/$h"; done
    for l in "${LIBS[@]}"; do $SUDO rm -f "$PREFIX/lib/$l"; done
    echo "KERNIX 삭제 완료: $PREFIX/include, $PREFIX/lib"
    exit 0
fi

if ! command -v clang >/dev/null || ! command -v make >/dev/null; then
    echo "오류: clang과 make가 필요합니다. 설치: xcode-select --install" >&2
    exit 1
fi

make -C "$SCRIPT_DIR" lib

$SUDO mkdir -p "$PREFIX/include" "$PREFIX/lib"
for h in "${HEADERS[@]}"; do $SUDO install -m 644 "$SCRIPT_DIR/include/$h" "$PREFIX/include/$h"; done
for l in "${LIBS[@]}"; do $SUDO install -m 644 "$SCRIPT_DIR/build/$l" "$PREFIX/lib/$l"; done

cat <<MSG
KERNIX 설치 완료
  헤더:       $PREFIX/include/{kernix.h,kernix_model.h}
  라이브러리: $PREFIX/lib/{libkernix.a,libkernix_metal.a}

빌드 예 (CPU + AMX):
  clang -O3 -std=c11 -I$PREFIX/include -DKERNIX_USE_ACCELERATE app.c -L$PREFIX/lib -lkernix -framework Accelerate -o app
빌드 예 (CPU + AMX + GPU, kernix.h를 쓰는 모든 소스에 -DKERNIX_USE_METAL):
  clang -O3 -std=c11 -I$PREFIX/include -DKERNIX_USE_ACCELERATE -DKERNIX_USE_METAL app.c -L$PREFIX/lib -lkernix -lkernix_metal \\
        -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation -o app
MSG
