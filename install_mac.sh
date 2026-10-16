#!/usr/bin/env bash

set -e

# KERNIX macOS Installer
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HEADER="$SCRIPT_DIR/include/kernix.h"
INSTALL_DIR="/usr/local/include"
INSTALL_FILE="$INSTALL_DIR/kernix.h"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: This installer supports macOS only."
    exit 1
fi

if [[ ! -f "$HEADER" ]]; then
    echo "Error: KERNIX header not found: $HEADER"
    exit 1
fi

sudo mkdir -p "$INSTALL_DIR"
sudo install -m 644 "$HEADER" "$INSTALL_FILE"

echo "KERNIX installed successfully!"
echo "Header: $INSTALL_FILE"
