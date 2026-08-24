#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

python3 tools/make_phase7g_presentation_freeze.py
python3 tools/validate_phase7g_presentation_freeze.py
