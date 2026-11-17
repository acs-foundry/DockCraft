#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"

mkdir -p "$BIN_DIR" "$APP_DIR" "$ICON_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/target/release/dockcraft" ]]; then
    install -m 755 "$REPO_ROOT/target/release/dockcraft" "$BIN_DIR/dockcraft"
fi
if [[ -f "$REPO_ROOT/target/release/dockcraft-cli" ]]; then
    install -m 755 "$REPO_ROOT/target/release/dockcraft-cli" "$BIN_DIR/dockcraft-cli"
fi
install -m 644 "$SCRIPT_DIR/dockcraft.desktop" "$APP_DIR/dockcraft.desktop"
install -m 644 "$SCRIPT_DIR/dockcraft.svg" "$ICON_DIR/dockcraft.svg"

echo "Installed DockCraft into $BIN_DIR"
