from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


def run_net_crossings(
    trajectory_jsonl_path: str,
    output_dir: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> bool:
    """Run net crossings extraction for a frame window."""
    if not trajectory_jsonl_path or not os.path.exists(trajectory_jsonl_path):
        return False
    os.makedirs(output_dir, exist_ok=True)

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "net_crossings_from_jsonl.py",
    )
    cmd = [
        sys.executable,
        script_path,
        "--trajectory-jsonl",
        trajectory_jsonl_path,
        "--output-dir",
        output_dir,
    ]
    if start_frame is not None:
        cmd.extend(["--start-frame", str(start_frame)])
    if end_frame is not None:
        cmd.extend(["--end-frame", str(end_frame)])

    result = subprocess.run(cmd, check=False)
    return result.returncode == 0
