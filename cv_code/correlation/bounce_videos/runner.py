from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


def run_bounce_videos(
    bounce_csv_path: str,
    camera: str,
    segments_dir: str,
    output_dir: str,
    triplet_csv_path: str = "",
    frames_before: int = 8,
    frames_after: int = 8,
    pause_frames: int = 4,
    limit: Optional[int] = None,
) -> bool:
    """Generate bounce clips for one camera."""
    if not bounce_csv_path or not os.path.exists(bounce_csv_path):
        return False
    if not segments_dir or not os.path.exists(segments_dir):
        return False

    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "bounce_clips_from_hls.py",
    )
    cmd = [
        sys.executable,
        script_path,
        "--bounce-csv",
        bounce_csv_path,
        "--camera",
        camera,
        "--segments-dir",
        segments_dir,
        "--output-dir",
        output_dir,
        "--frames-before",
        str(frames_before),
        "--frames-after",
        str(frames_after),
        "--pause-frames",
        str(pause_frames),
    ]
    if triplet_csv_path:
        cmd.extend(["--triplet-csv", triplet_csv_path])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, check=False)
    return result.returncode == 0
