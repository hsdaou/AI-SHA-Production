#!/usr/bin/env bash
#
# build_ros.sh — colcon build wrapper that works around the setuptools
# version conflict on this Jetson.
#
# Why this exists:
#   Humble's ament_python build needs an OLD setuptools (<= ~65), but the
#   NeMo / nv ML stack on this box pins a NEW setuptools (>= 70/79). A plain
#   `colcon build` while the new setuptools is active FAILS *and wipes the
#   installed package metadata* (.egg-info), which breaks the entry-point
#   wrappers (PackageNotFoundError at launch).
#
#   This script does the dance automatically:
#     1. records the setuptools version currently installed
#     2. pins a build-compatible setuptools (user-site only)
#     3. runs `colcon build` (passing through any extra args)
#     4. ALWAYS restores the original setuptools afterwards — even if the
#        build fails or you Ctrl-C — so the ML stack keeps working.
#
# Usage:
#   ./build_ros.sh                                 # build everything
#   ./build_ros.sh --packages-select stt_node      # one (or more) packages
#   ./build_ros.sh --packages-up-to robot_bringup  # a package and its deps
#   ./build_ros.sh --symlink-install ...           # any colcon args pass through
#
set -o pipefail

# setuptools version known-good for Humble ament_python builds.
BUILD_SETUPTOOLS="65.7.0"

# Find the colcon workspace root: walk up from this script's location until
# we hit a directory that contains a `src/` subdir. This lets the script live
# either at the workspace root (~/robot_ws) or inside the tracked repo
# (~/robot_ws/src) and still run colcon from the correct place.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT=""
_d="$_script_dir"
while [ "$_d" != "/" ]; do
    if [ -d "$_d/src" ]; then WS_ROOT="$_d"; break; fi
    _d="$(dirname "$_d")"
done
if [ -z "$WS_ROOT" ]; then
    echo "[build_ros] ERROR: could not find a colcon workspace (no src/ dir above $_script_dir)" >&2
    exit 1
fi
cd "$WS_ROOT" || { echo "[build_ros] cannot cd to $WS_ROOT" >&2; exit 1; }

current_setuptools() {
    python3 -c 'import setuptools; print(setuptools.__version__)' 2>/dev/null || true
}

ORIG_SETUPTOOLS="$(current_setuptools)"

restore_setuptools() {
    # Only act if we actually changed it.
    if [ -n "$ORIG_SETUPTOOLS" ] && [ "$(current_setuptools)" != "$ORIG_SETUPTOOLS" ]; then
        echo "[build_ros] Restoring setuptools ${ORIG_SETUPTOOLS} (for the ML stack) ..."
        if ! python3 -m pip install --user --quiet "setuptools==${ORIG_SETUPTOOLS}"; then
            echo "[build_ros] WARNING: could not restore setuptools ${ORIG_SETUPTOOLS}." >&2
            echo "[build_ros] Run manually: pip install --user 'setuptools==${ORIG_SETUPTOOLS}'" >&2
        fi
    fi
}
# Restore on ANY exit path (success, build failure, or Ctrl-C).
trap restore_setuptools EXIT

echo "[build_ros] workspace : $WS_ROOT"
echo "[build_ros] setuptools: ${ORIG_SETUPTOOLS:-unknown}  (build needs ${BUILD_SETUPTOOLS})"

if [ -z "$ORIG_SETUPTOOLS" ]; then
    echo "[build_ros] ERROR: could not detect setuptools via python3." >&2
    exit 1
fi

if [ "$ORIG_SETUPTOOLS" != "$BUILD_SETUPTOOLS" ]; then
    echo "[build_ros] Pinning setuptools ${BUILD_SETUPTOOLS} for the build ..."
    if ! python3 -m pip install --user --quiet "setuptools==${BUILD_SETUPTOOLS}"; then
        echo "[build_ros] ERROR: failed to install setuptools ${BUILD_SETUPTOOLS}." >&2
        exit 1
    fi
fi

# Base ROS 2 distro (sourcing the overlay is not needed to build).
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

echo "[build_ros] Running: colcon build $*"
colcon build "$@"
status=$?

if [ "$status" -eq 0 ]; then
    echo "[build_ros] ✓ build OK — setuptools ${ORIG_SETUPTOOLS} will be restored on exit."
else
    echo "[build_ros] ✗ build FAILED (exit ${status}) — setuptools ${ORIG_SETUPTOOLS} will still be restored." >&2
fi

exit $status
