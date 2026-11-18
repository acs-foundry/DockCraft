#!/usr/bin/env bash
set -euo pipefail

# Installs DockCraft for the current user, from either layout:
#   * the repository, where the binaries live in target/release/, or
#   * the released tarball, where they sit next to this script.

PREFIX="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_binary() {
    local name="$1"
    for candidate in "$SCRIPT_DIR/$name" "$REPO_ROOT/target/release/$name"; do
        if [[ -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

dockcraft_bin="$(find_binary dockcraft || true)"
cli_bin="$(find_binary dockcraft-cli || true)"

if [[ -z "$dockcraft_bin" && -z "$cli_bin" ]]; then
    echo "error: no DockCraft binaries found." >&2
    echo "       Build them first with: cargo build --release --workspace" >&2
    exit 1
fi

mkdir -p "$BIN_DIR" "$APP_DIR" "$ICON_DIR"

[[ -n "$dockcraft_bin" ]] && install -m 755 "$dockcraft_bin" "$BIN_DIR/dockcraft"
[[ -n "$cli_bin" ]] && install -m 755 "$cli_bin" "$BIN_DIR/dockcraft-cli"

install -m 644 "$SCRIPT_DIR/dockcraft.desktop" "$APP_DIR/dockcraft.desktop"
install -m 644 "$SCRIPT_DIR/dockcraft.svg" "$ICON_DIR/dockcraft.svg"

echo "Installed DockCraft into $BIN_DIR"
