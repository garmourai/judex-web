#!/usr/bin/env python3
"""
Concatenate bounce clip MP4s under bounce_clips/source and bounce_clips/sink
(defaults below). Uses ffmpeg concat demuxer (-c copy). Requires ffmpeg on PATH.

Usage:
  python concat_bounce_mp4s.py
  python concat_bounce_mp4s.py --source-dir /path --sink-dir /path2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile


def _list_mp4s(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    names = [n for n in os.listdir(directory) if n.lower().endswith(".mp4")]
    return [os.path.join(directory, n) for n in sorted(names)]


def _concat_ffmpeg(paths: list[str], output_path: str) -> None:
    if not paths:
        print(f"[concat_bounce_mp4s] skip (no mp4): {output_path!r}", file=sys.stderr)
        return
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # concat demuxer: one line per file; escape single quotes in path for ffmpeg
    lines = []
    for p in paths:
        ap = os.path.abspath(p)
        ap = ap.replace("'", "'\\''")
        lines.append(f"file '{ap}'")
    fd, list_path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            output_path,
        ]
        print(f"[concat_bounce_mp4s] writing {len(paths)} clips -> {output_path}", flush=True)
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate bounce MP4s per directory via ffmpeg.")
    p.add_argument(
        "--source-dir",
        default="/mnt/data/cv_output/bounce_clips/source",
        help="Directory with source-camera bounce MP4s",
    )
    p.add_argument(
        "--sink-dir",
        default="/mnt/data/cv_output/bounce_clips/sink",
        help="Directory with sink-camera bounce MP4s",
    )
    p.add_argument(
        "--out-source",
        default="/mnt/data/cv_output/bounce_clips/merged_source.mp4",
        help="Output path for concatenated source clips",
    )
    p.add_argument(
        "--out-sink",
        default="/mnt/data/cv_output/bounce_clips/merged_sink.mp4",
        help="Output path for concatenated sink clips",
    )
    args = p.parse_args()

    src = _list_mp4s(args.source_dir)
    snk = _list_mp4s(args.sink_dir)
    if not src:
        print(f"[concat_bounce_mp4s] no .mp4 in {args.source_dir}", file=sys.stderr)
    else:
        _concat_ffmpeg(src, args.out_source)
    if not snk:
        print(f"[concat_bounce_mp4s] no .mp4 in {args.sink_dir}", file=sys.stderr)
    else:
        _concat_ffmpeg(snk, args.out_sink)


if __name__ == "__main__":
    main()
