#!/usr/bin/env bash
# install_mac.sh와 같음 (예전 이름 호환). 인자와 PREFIX도 그대로 넘김
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_mac.sh" "$@"
