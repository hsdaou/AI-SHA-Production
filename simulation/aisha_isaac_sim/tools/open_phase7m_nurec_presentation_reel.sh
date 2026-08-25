#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="$PACKAGE_ROOT/media/videos/AI-SHA_Phase7M_NuRec_Principal_Visit_Presentation.mp4"
cd "$PACKAGE_ROOT"

python3 tools/validate_phase7m_nurec_presentation.py
[[ -f "$VIDEO" ]] || { echo "missing local Phase 7M reel: $VIDEO" >&2; exit 2; }
xdg-open "$VIDEO" >/dev/null 2>&1 &
