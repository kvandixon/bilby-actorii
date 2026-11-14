#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chmod +x "$REPO_ROOT/scripts/release_gate.py" "$REPO_ROOT/scripts/hooks/pre-push"
ln -sf "../../scripts/hooks/pre-push" "$REPO_ROOT/.git/hooks/pre-push"
echo "Installed .git/hooks/pre-push -> scripts/hooks/pre-push"
