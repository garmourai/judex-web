"""
Resolve calibration outputs under ``tools/pickleball_calib`` (e.g. ``calibration_1512``).

If multiple calibration subfolders exist, the one with the newest
``source/camera_object.pkl`` (or ``camera_object.yaml``) wins.

Override the directory with env ``JUDEX_CALIB_DATA_DIR``.
"""

from __future__ import annotations

import os
from typing import List, Tuple


def repo_root() -> str:
    """judex-web root: .../cv_code/shot_segmentation_and_landing/calib_paths.py -> two levels up."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def pickleball_calib_dir() -> str:
    return os.path.join(repo_root(), "tools", "pickleball_calib")


def _calibration_mtime(calibration_dir: str) -> float:
    """Best timestamp for "how fresh" this calib bundle is."""
    pkl = os.path.join(calibration_dir, "source", "camera_object.pkl")
    yml = os.path.join(calibration_dir, "source", "camera_object.yaml")
    if os.path.isfile(pkl):
        return os.path.getmtime(pkl)
    if os.path.isfile(yml):
        return os.path.getmtime(yml)
    return 0.0


def get_latest_calib_data_dir() -> str:
    """
    Return absolute path to the latest calibration data directory under
    ``tools/pickleball_calib``.

    Scans immediate subdirectories; each must contain ``source/camera_object.pkl``
    or ``source/camera_object.yaml``. Picks the directory with newest mtime.

    Falls back to ``tools/pickleball_calib/calibration_1512`` if none qualify.
    """
    env = os.environ.get("JUDEX_CALIB_DATA_DIR", "").strip()
    if env:
        return os.path.abspath(env)

    root = pickleball_calib_dir()
    fallback = os.path.join(root, "calibration_1512")
    if not os.path.isdir(root):
        return fallback

    candidates: List[Tuple[float, str]] = []
    for name in os.listdir(root):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        pkl = os.path.join(sub, "source", "camera_object.pkl")
        yml = os.path.join(sub, "source", "camera_object.yaml")
        if os.path.isfile(pkl) or os.path.isfile(yml):
            candidates.append((_calibration_mtime(sub), sub))

    if not candidates:
        return fallback

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def default_source_camera_pkl_path() -> str:
    """Default reprojection pickle for the **source** camera (overlay / 3D lines)."""
    return os.path.join(get_latest_calib_data_dir(), "source", "camera_object.pkl")


def default_sink_camera_pkl_path() -> str:
    """Sink camera pickle path (pairwise / dual-camera tools)."""
    return os.path.join(get_latest_calib_data_dir(), "sink", "camera_object.pkl")
