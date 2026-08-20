#!/usr/bin/env python3
"""Write a reproducible local Isaac workstation inventory."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "workstation_inventory.json"


def command(args: list[str]) -> dict[str, object]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False, "command": args}
    completed = subprocess.run(
        [executable, *args[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {
        "available": True,
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def os_release() -> dict[str, str]:
    values = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip('"')
    return values


def isaac_installations() -> list[dict[str, object]]:
    roots = []
    downloads = Path.home() / "Downloads"
    for version_file in sorted(downloads.glob("isaac-sim-*/VERSION")):
        root = version_file.parent
        roots.append(
            {
                "root": str(root),
                "version": version_file.read_text(encoding="utf-8").strip(),
                "python": str(root / "python.sh") if (root / "python.sh").exists() else None,
                "launcher": str(root / "isaac-sim.sh") if (root / "isaac-sim.sh").exists() else None,
            }
        )
    return roots


def main() -> int:
    ros = [path.name for path in sorted(Path("/opt/ros").glob("*")) if path.is_dir()]
    isaac_lab = [str(path) for path in sorted((Path.home() / "Downloads").glob("**/isaaclab.sh"))]
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "os_release": os_release(),
        "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "isaac_sim": isaac_installations(),
        "isaac_lab": {"found": bool(isaac_lab), "launchers": isaac_lab},
        "ros_distros": ros,
        "ros2_executable": shutil.which("ros2"),
        "display": {
            "session_type": os.environ.get("XDG_SESSION_TYPE"),
            "display": os.environ.get("DISPLAY"),
            "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
