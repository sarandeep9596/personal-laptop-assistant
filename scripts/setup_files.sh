#!/usr/bin/env bash
# Create required runtime files that are not tracked in git.
# Run once after cloning: bash scripts/setup_files.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$ROOT/Frontend/Files"

python3 - <<'EOF'
import os, pathlib

root = pathlib.Path(os.environ.get("ROOT", "."))
files = {
    "Frontend/Files/Mic.data":       "False",
    "Frontend/Files/Status.data":    "False",
    "Frontend/Files/Responses.data": "",
    "Frontend/Files/Database.data":  "",
}
for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(content, encoding="utf-8")
        print(f"  created {rel}")
    else:
        print(f"  exists  {rel}")
EOF

echo "Setup complete."
