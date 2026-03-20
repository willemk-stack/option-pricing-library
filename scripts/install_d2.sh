#!/usr/bin/env bash

set -euo pipefail

D2_VERSION="${1:-${D2_VERSION:-v0.7.1}}"
D2_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$D2_TMP_DIR"' EXIT

mkdir -p "$HOME/.local/bin"
curl -fsSLo "$D2_TMP_DIR/d2.tgz" "https://github.com/terrastruct/d2/releases/download/${D2_VERSION}/d2-${D2_VERSION}-linux-amd64.tar.gz"
tar -xzf "$D2_TMP_DIR/d2.tgz" -C "$D2_TMP_DIR"

D2_BIN="$(find "$D2_TMP_DIR" -type f -path '*/bin/d2' -print -quit)"
test -x "$D2_BIN"
install "$D2_BIN" "$HOME/.local/bin/d2"
echo "$HOME/.local/bin" >> "$GITHUB_PATH"
"$HOME/.local/bin/d2" --version