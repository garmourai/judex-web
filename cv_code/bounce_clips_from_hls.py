#!/usr/bin/env python3
"""
Generate short MP4 clips (300×300) around bounce points from HLS (playlist.m3u8 + seg_*.ts).

Reads frames via bounce_events.csv (bbox_*) using bbox center for crop anchor.
Source: frame_id == global source index. Sink: requires triplet CSV mapping
Source_Index -> Sink_Index.

Requires hls_segment_frame_index.csv. By default the manifest directory matches the
triplet pipeline: /mnt/data/cv_output/reader/source or .../reader/sink (see
DEFAULT_READER_MANIFEST_*). Override with --manifest-dir (e.g. put the CSV next to
playlist.m3u8 by setting --manifest-dir to the same path as --segments-dir).
This script does not import m3u8_reader or hls_segment_manifest.

Default `--camera both` writes clips under `--output-dir/source/` and `.../sink/`.
Rows are sorted by bounce_frame before decode.

Runs indefinitely: polls `bounce_events.csv` every 2 seconds (override with
`--poll-interval`), encodes any new rows since the last run. First run processes
all rows if no state file exists. `--state-file` (default under `--output-dir`)
stores `next_row_index` for resume after restart. Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BBox2D = Dict[str, float]

# Ctrl+C: OpenCV spends long stretches in C extensions (VideoCapture.read / VideoWriter.write),
# so the default SIGINT -> KeyboardInterrupt path often does not run until those return.
# We install a handler that sets a flag; code checks it between frames/rows. A second Ctrl+C
# restores the default handler and re-delivers SIGINT for an immediate exit.

_shutdown_requested = False


def _sigint_handler(signum: int, frame: object) -> None:
    global _shutdown_requested
    if _shutdown_requested:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
        return
    _shutdown_requested = True
    try:
        print(
            "\n[bounce_clips_from_hls] Ctrl+C: stopping soon (second Ctrl+C forces exit)...",
            flush=True,
            file=sys.stderr,
        )
    except Exception:
        pass


def _check_shutdown() -> None:
    if _shutdown_requested:
        raise KeyboardInterrupt


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in short slices so Ctrl+C is noticed quickly (custom SIGINT handler)."""
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while True:
        _check_shutdown()
        now = time.monotonic()
        if now >= end:
            return
        time.sleep(min(0.2, end - now))


MANIFEST_FILENAME = "hls_segment_frame_index.csv"

CROP_MAX = 300  # max side length; actual crop may be smaller near borders

# ---------------------------------------------------------------------------
# Default paths (single place to edit for your deployment)
# ---------------------------------------------------------------------------
_DATA_ROOT = "/mnt/data"
_CV_OUTPUT = f"{_DATA_ROOT}/cv_output"
_SYNC_REPORTS = f"{_DATA_ROOT}/mar30_test/sync_reports"
_SEG_1547 = f"{_SYNC_REPORTS}/segments_1547/sync"

DEFAULT_BOUNCE_CSV = f"{_CV_OUTPUT}/correlation/bounce_events.csv"
DEFAULT_CAMERA = "both"
DEFAULT_SEGMENTS_DIR_SOURCE = f"{_SYNC_REPORTS}/ts_segments_source/1547"
DEFAULT_SEGMENTS_DIR_SINK = f"{_SYNC_REPORTS}/ts_segments_sink/1547"
# Same layout as triplet pipeline (unique_output_dir/reader/source|sink/hls_segment_frame_index.csv)
DEFAULT_READER_MANIFEST_SOURCE = f"{_CV_OUTPUT}/reader/source"
DEFAULT_READER_MANIFEST_SINK = f"{_CV_OUTPUT}/reader/sink"
DEFAULT_TRIPLET_CSV = f"{_SEG_1547}/hls_sync_1547_triple.csv"
DEFAULT_OUTPUT_DIR = f"{_CV_OUTPUT}/bounce_clips"
DEFAULT_STATE_FILE = ""  # empty => <output-dir>/bounce_clips_cursor.json


def _load_segment_frame_index_for_bounce(manifest_dir: str) -> Tuple[List[int], List[int]]:
    """
    Load cumulative_offsets and frame_counts from hls_segment_frame_index.csv
    with the same validation rules as cv_code.hls_segment_manifest.load_segment_manifest.
    """
    path = os.path.join(manifest_dir, MANIFEST_FILENAME)
    if not os.path.isfile(path):
        raise SystemExit(
            f"Missing {MANIFEST_FILENAME!r} under {manifest_dir!r}. "
            "Run the triplet CV pipeline (writes under unique_output_dir/reader/source|sink), "
            "or set --manifest-dir to the folder that contains the index CSV."
        )
    by_seg: Dict[int, Tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Empty or headerless manifest: {path!r}")
        for row in reader:
            try:
                si = int(row["segment_index"])
                cs = int(row["cumulative_start_frame"])
                fc = int(row["frame_count"])
            except (KeyError, ValueError, TypeError):
                continue
            by_seg[si] = (cs, fc)
    if not by_seg:
        raise SystemExit(f"No valid rows in manifest: {path!r}")
    max_seg = max(by_seg.keys())
    cumulative_offsets: List[int] = []
    frame_counts: List[int] = []
    for i in range(max_seg + 1):
        if i not in by_seg:
            raise SystemExit(
                f"Manifest {path!r} is not contiguous (missing segment_index {i})."
            )
        cs, fc = by_seg[i]
        cumulative_offsets.append(cs)
        frame_counts.append(fc)
    if cumulative_offsets[0] != 0:
        raise SystemExit(
            f"Manifest {path!r}: cumulative_start_frame for segment 0 must be 0."
        )
    for i in range(len(cumulative_offsets) - 1):
        if cumulative_offsets[i] + frame_counts[i] != cumulative_offsets[i + 1]:
            raise SystemExit(
                f"Manifest {path!r}: offset chain broken between segment {i} and {i + 1}."
            )
    return cumulative_offsets, frame_counts


class BounceHlsFrameReader:
    """Decode frames from seg_*.ts using hls_segment_frame_index.csv (no m3u8_reader)."""

    def __init__(self, segments_dir: str, manifest_dir: Optional[str] = None) -> None:
        self._segments_dir = segments_dir
        md = manifest_dir if manifest_dir is not None else segments_dir
        loaded = _load_segment_frame_index_for_bounce(md)
        self._cumulative_offsets: List[int] = list(loaded[0])
        self._frame_counts: List[int] = list(loaded[1])
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_seg_idx: int = -1
        self._current_seg_pos: int = 0
        self._orig_width: Optional[int] = None
        self._orig_height: Optional[int] = None

    def _segment_ts_path(self, seg_idx: int) -> str:
        return os.path.join(self._segments_dir, f"seg_{seg_idx:05d}.ts")

    def _find_segment(self, global_frame_index: int) -> Optional[int]:
        for seg_idx in range(len(self._cumulative_offsets)):
            start = self._cumulative_offsets[seg_idx]
            end = start + self._frame_counts[seg_idx] - 1
            if start <= global_frame_index <= end:
                return seg_idx
        return None

    def read_frame_at(self, global_frame_index: int) -> Optional[np.ndarray]:
        seg_idx = self._find_segment(global_frame_index)
        if seg_idx is None:
            last_end = (
                self._cumulative_offsets[-1] + self._frame_counts[-1] - 1
                if self._cumulative_offsets
                else -1
            )
            print(
                f"[bounce_clips_from_hls] global_frame={global_frame_index} out of manifest range "
                f"(last_frame_index={last_end})"
            )
            return None
        local_offset = global_frame_index - self._cumulative_offsets[seg_idx]

        if self._current_seg_idx != seg_idx:
            ts_path = self._segment_ts_path(seg_idx)
            if not os.path.isfile(ts_path):
                print(f"[bounce_clips_from_hls] missing segment file: {ts_path}")
                return None
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            _check_shutdown()
            cap = cv2.VideoCapture(ts_path)
            if not cap.isOpened():
                print(f"[bounce_clips_from_hls] failed to open: {ts_path}")
                return None
            if self._orig_width is None:
                self._orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._cap = cap
            self._current_seg_idx = seg_idx
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, local_offset)
            self._current_seg_pos = local_offset
        else:
            if local_offset > self._current_seg_pos:
                for _ in range(local_offset - self._current_seg_pos):
                    self._cap.grab()
                self._current_seg_pos = local_offset
            elif local_offset < self._current_seg_pos:
                assert self._cap is not None
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, local_offset)
                self._current_seg_pos = local_offset

        assert self._cap is not None
        _check_shutdown()
        ret, frame = self._cap.read()
        if not ret:
            return None
        self._current_seg_pos = local_offset + 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._current_seg_idx = -1


def _apply_default_paths(args: argparse.Namespace) -> None:
    """Fill None CLI args from defaults; pick source vs sink segments dir by camera."""
    if getattr(args, "bounce_csv", None) is None:
        args.bounce_csv = DEFAULT_BOUNCE_CSV
    if getattr(args, "output_dir", None) is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    if getattr(args, "triplet_csv", None) is None:
        args.triplet_csv = DEFAULT_TRIPLET_CSV
    if args.camera == "both":
        # Per-stream paths are chosen in main(); leave segments_dir / manifest_dir unset.
        return
    if getattr(args, "segments_dir", None) is None:
        args.segments_dir = (
            DEFAULT_SEGMENTS_DIR_SINK
            if args.camera == "sink"
            else DEFAULT_SEGMENTS_DIR_SOURCE
        )
    if getattr(args, "manifest_dir", None) is None:
        args.manifest_dir = (
            DEFAULT_READER_MANIFEST_SINK
            if args.camera == "sink"
            else DEFAULT_READER_MANIFEST_SOURCE
        )


def _segments_and_manifest_for_camera(camera: str) -> Tuple[str, str]:
    """Default HLS dir and reader manifest dir for one stream (matches triplet layout)."""
    if camera == "sink":
        return DEFAULT_SEGMENTS_DIR_SINK, DEFAULT_READER_MANIFEST_SINK
    return DEFAULT_SEGMENTS_DIR_SOURCE, DEFAULT_READER_MANIFEST_SOURCE


def _bbox_from_bounce_row(row: Dict[str, str], camera: str) -> Optional[BBox2D]:
    prefix = "bbox_source_" if camera == "source" else "bbox_sink_"
    sx = row.get(prefix + "x", "")
    sy = row.get(prefix + "y", "")
    sw = row.get(prefix + "w", "")
    sh = row.get(prefix + "h", "")
    if not sx or not sy or not sw or not sh:
        return None
    return {
        "x": float(sx),
        "y": float(sy),
        "w": float(sw),
        "h": float(sh),
    }


def _load_triplet_source_to_sink(path: str) -> Dict[int, int]:
    """Build Source_Index -> Sink_Index for rows where both are present."""
    mapping: Dict[int, int] = {}
    if not path or not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                si_s = row.get("Source_Index", "").strip()
                sk_s = row.get("Sink_Index", "").strip()
                if not si_s:
                    continue
                si = int(si_s)
                if not sk_s or sk_s.lower() in ("none", "null", "nan"):
                    continue
                mapping[si] = int(sk_s)
            except (ValueError, TypeError):
                continue
    return mapping


def _square_crop_rect(cx: float, cy: float, w: int, h: int, max_side: int = CROP_MAX) -> Tuple[int, int, int, int]:
    """
    Largest axis-aligned square centered on (cx, cy), side at most max_side, clipped to image.
    Returns (x0, y0, x1, y1) with x1, y1 exclusive for numpy slicing.
    """
    max_half = max_side // 2
    xi = int(round(cx))
    yi = int(round(cy))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    ext_l = xi
    ext_r = w - 1 - xi
    ext_t = yi
    ext_b = h - 1 - yi
    half = min(max_half, ext_l, ext_r, ext_t, ext_b)
    half = max(0, half)
    x0 = xi - half
    x1 = xi + half + 1
    y0 = yi - half
    y1 = yi + half + 1
    return (x0, y0, x1, y1)


def _crop_with_rect(img, rect: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = rect
    return img[y0:y1, x0:x1]


def _draw_landing_bbox_from_xywh(
    patch,
    rect: Tuple[int, int, int, int],
    bbox_xywh: Optional[BBox2D],
):
    """
    Draw bbox from full-image xywh mapped into crop-local coords.
    If bbox is missing/invalid, return patch unchanged.
    """
    if not bbox_xywh:
        return patch

    x0, y0, _, _ = rect
    bw = float(bbox_xywh["w"])
    bh = float(bbox_xywh["h"])
    print(f"[bounce_clips_from_hls] landing bbox size: w={bw:.2f}px h={bh:.2f}px")
    bx = float(bbox_xywh["x"]) - float(x0)
    by = float(bbox_xywh["y"]) - float(y0)

    out = patch.copy()
    ph, pw = out.shape[:2]
    rx1 = max(0, int(round(bx)))
    ry1 = max(0, int(round(by)))
    rx2 = min(pw - 1, int(round(bx + bw)))
    ry2 = min(ph - 1, int(round(by + bh)))
    if rx2 <= rx1 or ry2 <= ry1:
        return patch
    cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), thickness=2)
    return out


def _crop_landing_bbox_region(
    patch,
    rect: Tuple[int, int, int, int],
    bbox_xywh: Optional[BBox2D],
) -> Optional[object]:
    """Return exact landing bbox pixel crop (patch-local), or None if unavailable/invalid."""
    if not bbox_xywh:
        return None
    x0, y0, _, _ = rect
    bx = float(bbox_xywh["x"]) - float(x0)
    by = float(bbox_xywh["y"]) - float(y0)
    bw = float(bbox_xywh["w"])
    bh = float(bbox_xywh["h"])

    ph, pw = patch.shape[:2]
    rx1 = max(0, int(round(bx)))
    ry1 = max(0, int(round(by)))
    rx2 = min(pw, int(round(bx + bw)))
    ry2 = min(ph, int(round(by + bh)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return patch[ry1:ry2, rx1:rx2]


def _fps_from_segments_dir(segments_dir: str) -> float:
    ts0 = os.path.join(segments_dir, "seg_00000.ts")
    if os.path.exists(ts0):
        cap = cv2.VideoCapture(ts0)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            if fps > 0:
                return fps
    return 30.0


def _load_bounce_csv_rows(path: str) -> List[Tuple[int, Dict[str, str]]]:
    """Return (row_index, row) for each data row after the header (0-based)."""
    out: List[Tuple[int, Dict[str, str]]] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            out.append((i, row))
    return out


def _load_cursor_state(path: str) -> int:
    """Next row_index to process (0 = start of file)."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return max(0, int(d.get("next_row_index", 0)))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0


def _save_cursor_state(path: str, next_row_index: int) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"next_row_index": next_row_index}, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _resolve_read_indices(
    camera: str,
    source_frame_ids: List[int],
    triplet_map: Dict[int, int],
) -> Optional[List[int]]:
    """Return list of global indices for BounceHlsFrameReader.read_frame_at, or None if invalid."""
    if camera == "source":
        return list(source_frame_ids)
    out: List[int] = []
    for fid in source_frame_ids:
        sk = triplet_map.get(fid)
        if sk is None:
            return None
        out.append(sk)
    # Reader requires monotonic non-decreasing read indices in time order
    for i in range(len(out) - 1):
        if out[i] > out[i + 1]:
            return None
    return out


def _make_clip(
    reader: BounceHlsFrameReader,
    bounce_frame: int,
    source_frame_ids: List[int],
    read_indices: List[int],
    out_path: str,
    source_fps: float,
    playback_speed: float,
    pause_frames: int,
    landing_bbox: Optional[BBox2D],
    landing_bbox_image_path: str,
) -> Tuple[bool, Dict[str, float]]:
    """
    One crop rect from bounce_frame anchor; same rect applied to every frame.
    On the landing frame (bounce_frame), duplicate that patch `pause_frames` times for a hold
    (bbox overlay / bbox PNG export disabled).
    output_fps = source_fps * playback_speed (e.g. 0.5 = half-speed playback).

    Returns (success, profile) with seconds: decode_s, crop_s, write_s, total_s.
    """
    t0 = time.perf_counter()

    def _prof(decode_s: float, crop_s: float, write_s: float) -> Dict[str, float]:
        total_s = time.perf_counter() - t0
        return {
            "decode_s": decode_s,
            "crop_s": crop_s,
            "write_s": write_s,
            "total_s": total_s,
        }

    assert len(source_frame_ids) == len(read_indices)
    frames: List[object] = []
    t_dec = time.perf_counter()
    for sfid, ridx in zip(source_frame_ids, read_indices):
        frame = reader.read_frame_at(ridx)
        if frame is None:
            print(f"[bounce_clips_from_hls] decode failed at read_index={ridx} (source_frame={sfid})")
            decode_s = time.perf_counter() - t_dec
            return False, _prof(decode_s, 0.0, 0.0)
        frames.append(frame)
    decode_s = time.perf_counter() - t_dec

    ih, iw = frames[0].shape[:2]
    if not landing_bbox:
        print(f"[bounce_clips_from_hls] skip bounce_frame={bounce_frame}: missing bbox for camera")
        return False, _prof(decode_s, 0.0, 0.0)
    t_crop = time.perf_counter()
    u = float(landing_bbox["x"]) + float(landing_bbox["w"]) / 2.0
    v = float(landing_bbox["y"]) + float(landing_bbox["h"]) / 2.0
    rect = _square_crop_rect(u, v, iw, ih, CROP_MAX)

    patches: List[object] = []
    for frame in frames:
        patches.append(_crop_with_rect(frame, rect))

    if not patches:
        crop_s = time.perf_counter() - t_crop
        return False, _prof(decode_s, crop_s, 0.0)

    try:
        landing_idx = source_frame_ids.index(bounce_frame)
    except ValueError:
        landing_idx = -1
        print(f"[bounce_clips_from_hls] warning: bounce_frame {bounce_frame} not in window, no bbox/pause")

    out_h, out_w = patches[0].shape[:2]
    out_fps = max(1e-6, float(source_fps) * float(playback_speed))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    crop_s = time.perf_counter() - t_crop

    t_wr = time.perf_counter()
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"[bounce_clips_from_hls] could not open VideoWriter for {out_path}")
        write_s = time.perf_counter() - t_wr
        return False, _prof(decode_s, crop_s, write_s)

    n_hold = max(1, int(pause_frames)) if landing_idx >= 0 else 0

    for i, p in enumerate(patches):
        _check_shutdown()
        if i == landing_idx and landing_idx >= 0:
            # p_marked = _draw_landing_bbox_from_xywh(p, rect, landing_bbox)
            # bbox_crop = _crop_landing_bbox_region(p, rect, landing_bbox)
            # if bbox_crop is not None:
            #     cv2.imwrite(landing_bbox_image_path, bbox_crop)
            for _ in range(n_hold):
                _check_shutdown()
                writer.write(p)
        else:
            writer.write(p)

    writer.release()
    write_s = time.perf_counter() - t_wr
    return True, _prof(decode_s, crop_s, write_s)


def _process_rows_with_reader(
    rows_with_idx: List[Tuple[int, Dict[str, str]]],
    reader: BounceHlsFrameReader,
    *,
    camera: str,
    triplet_map: Dict[int, int],
    source_fps: float,
    playback_speed: float,
    frames_before: int,
    frames_after: int,
    pause_frames: int,
    output_dir: str,
) -> Tuple[int, int]:
    """Process rows in bounce_frame order; returns (n_ok, n_skip)."""

    def _sort_key(item: Tuple[int, Dict[str, str]]) -> Tuple[int, int]:
        row_index, row = item
        try:
            bf = int(row["bounce_frame"])
        except (KeyError, ValueError):
            bf = 1 << 30
        return (bf, row_index)

    rows_sorted = sorted(rows_with_idx, key=_sort_key)
    n_ok = 0
    n_skip = 0

    for row_index, row in rows_sorted:
        _check_shutdown()
        try:
            bounce_frame = int(row["bounce_frame"])
        except (KeyError, ValueError) as e:
            print(f"[bounce_clips_from_hls] skip row {row_index}: {e}")
            n_skip += 1
            continue

        landing_bbox = _bbox_from_bounce_row(row, camera)
        if landing_bbox is not None:
            print(
                f"[bounce_clips_from_hls] bounce_frame={bounce_frame} "
                f"bbox x={landing_bbox['x']:.2f} y={landing_bbox['y']:.2f} "
                f"w={landing_bbox['w']:.2f} h={landing_bbox['h']:.2f}"
            )
        else:
            print(
                f"[bounce_clips_from_hls] bounce_frame={bounce_frame} "
                f"bbox missing for camera={camera}"
            )
        start_f = bounce_frame - frames_before
        end_f = bounce_frame + frames_after
        source_frame_ids = list(range(start_f, end_f + 1))

        read_indices = _resolve_read_indices(camera, source_frame_ids, triplet_map)
        if read_indices is None:
            print(
                f"[bounce_clips_from_hls] skip bounce_frame={bounce_frame}: "
                f"missing or non-monotonic sink_index for frames in [{start_f}, {end_f}]"
            )
            n_skip += 1
            continue

        out_name = f"bounce_{bounce_frame}_{row_index:05d}.mp4"
        out_path = os.path.join(output_dir, out_name)
        bbox_img_path = os.path.join(output_dir, f"bounce_{bounce_frame}_{row_index:05d}_bbox.png")

        ok, prof = _make_clip(
            reader=reader,
            bounce_frame=bounce_frame,
            source_frame_ids=source_frame_ids,
            read_indices=read_indices,
            out_path=out_path,
            source_fps=source_fps,
            playback_speed=playback_speed,
            pause_frames=pause_frames,
            landing_bbox=landing_bbox,
            landing_bbox_image_path=bbox_img_path,
        )
        d, c, w, tot = (
            prof["decode_s"],
            prof["crop_s"],
            prof["write_s"],
            prof["total_s"],
        )

        if ok:
            if bounce_frame in source_frame_ids:
                n_out = len(source_frame_ids) - 1 + max(1, pause_frames)
            else:
                n_out = len(source_frame_ids)
            print(
                f"[bounce_clips_from_hls] clip profile decode={d:.3f}s crop={c:.3f}s "
                f"write={w:.3f}s total={tot:.3f}s | bounce_frame={bounce_frame} "
                f"row={row_index} -> {out_path} ({n_out} output frames)"
            )
            n_ok += 1
        else:
            print(
                f"[bounce_clips_from_hls] clip profile decode={d:.3f}s crop={c:.3f}s "
                f"write={w:.3f}s total={tot:.3f}s | bounce_frame={bounce_frame} "
                f"row={row_index} (failed)"
            )
            n_skip += 1

    return n_ok, n_skip


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bounce clips from HLS: crop up to 300×300 centered on bounce_frame; optional slow-mo."
    )
    p.add_argument(
        "--bounce-csv",
        default=None,
        help=f"bounce_events.csv (default: {DEFAULT_BOUNCE_CSV})",
    )
    p.add_argument(
        "--camera",
        choices=("source", "sink", "both"),
        default=DEFAULT_CAMERA,
        help=(
            "Process source, sink, or both (default: both). "
            "For both: clips under output-dir/source/ and output-dir/sink/. "
            "Requires --triplet-csv for both and sink."
        ),
    )
    p.add_argument(
        "--segments-dir",
        default=None,
        help=(
            f"HLS dir with seg_*.ts; default source={DEFAULT_SEGMENTS_DIR_SOURCE} "
            f"sink={DEFAULT_SEGMENTS_DIR_SINK}"
        ),
    )
    p.add_argument(
        "--manifest-dir",
        default=None,
        help=(
            f"Directory containing hls_segment_frame_index.csv (default: triplet reader paths "
            f"{DEFAULT_READER_MANIFEST_SOURCE} or {DEFAULT_READER_MANIFEST_SINK} by --camera). "
            "Use the same path as --segments-dir if the index lives next to playlist.m3u8."
        ),
    )
    p.add_argument(
        "--triplet-csv",
        default=None,
        help=f"triple CSV for --camera sink or both (default: {DEFAULT_TRIPLET_CSV})",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=f"output MP4 directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override source video FPS (default: from first .ts or 30); output FPS = this × playback-speed",
    )
    p.add_argument(
        "--playback-speed",
        type=float,
        default=0.5,
        help="multiply source FPS for output (default 0.5 = half-speed playback)",
    )
    p.add_argument("--frames-before", type=int, default=5, help="frames before bounce_frame (default 5)")
    p.add_argument("--frames-after", type=int, default=5, help="frames after bounce_frame (default 5)")
    p.add_argument(
        "--pause-frames",
        type=int,
        default=5,
        help="how many frames to hold at landing (same frame + bbox repeated; default 5)",
    )
    p.add_argument("--limit", type=int, default=None, help="max number of bounce rows to process per batch")
    p.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Seconds between CSV polls (default: 2)",
    )
    p.add_argument(
        "--state-file",
        default="",
        help="JSON file storing next_row_index (default: <output-dir>/bounce_clips_cursor.json)",
    )
    args = p.parse_args()
    _apply_default_paths(args)

    if args.camera in ("sink", "both") and not (args.triplet_csv or "").strip():
        raise SystemExit("--triplet-csv is required when --camera sink or both")
    if args.frames_before < 0 or args.frames_after < 0:
        raise SystemExit("--frames-before and --frames-after must be non-negative")
    if args.playback_speed <= 0:
        raise SystemExit("--playback-speed must be positive")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")

    os.makedirs(args.output_dir, exist_ok=True)

    state_path = args.state_file or os.path.join(args.output_dir, "bounce_clips_cursor.json")

    triplet_map = _load_triplet_source_to_sink(args.triplet_csv or "")

    # (camera_name, segments_dir, manifest_dir, clip_output_dir)
    if args.camera == "both":
        if args.segments_dir is not None or args.manifest_dir is not None:
            print(
                "[bounce_clips_from_hls] note: --camera both uses built-in source/sink "
                "segment and manifest paths; ignoring --segments-dir / --manifest-dir",
                flush=True,
            )
        seg_s, man_s = _segments_and_manifest_for_camera("source")
        seg_k, man_k = _segments_and_manifest_for_camera("sink")
        streams: List[Tuple[str, str, str, str]] = [
            ("source", seg_s, man_s, os.path.join(args.output_dir, "source")),
            ("sink", seg_k, man_k, os.path.join(args.output_dir, "sink")),
        ]
    else:
        streams = [
            (args.camera, args.segments_dir, args.manifest_dir, args.output_dir),
        ]

    _shutdown_requested = False
    prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        state_on_disk = os.path.exists(state_path)
        print(
            f"[bounce_clips_from_hls] polling {args.bounce_csv!r} every {args.poll_interval}s "
            f"(Ctrl+C to stop) | state {state_path!r} "
            f"({'resume' if state_on_disk else 'first run: encode all rows in CSV, then new rows only'})",
            flush=True,
        )
        while True:
            _check_shutdown()
            all_rows = _load_bounce_csv_rows(args.bounce_csv)
            cursor = _load_cursor_state(state_path)
            if not state_on_disk:
                pending = list(all_rows)
                if args.limit is not None:
                    pending = pending[: args.limit]
            else:
                pending = [(i, r) for i, r in all_rows if i >= cursor]
                if args.limit is not None:
                    pending = pending[: args.limit]

            if pending:
                batch_ok = batch_skip = 0
                for cam, seg, man, out in streams:
                    os.makedirs(out, exist_ok=True)
                    source_fps = args.fps if args.fps is not None else _fps_from_segments_dir(seg)
                    manifest_csv = os.path.join(man, MANIFEST_FILENAME)
                    print(
                        f"[bounce_clips_from_hls] camera={cam} segments_dir={seg!r} "
                        f"manifest_csv={manifest_csv!r} clip_output_dir={out!r}",
                        flush=True,
                    )
                    reader = BounceHlsFrameReader(seg, manifest_dir=man)
                    try:
                        n_ok, n_skip = _process_rows_with_reader(
                            pending,
                            reader,
                            camera=cam,
                            triplet_map=triplet_map,
                            source_fps=source_fps,
                            playback_speed=args.playback_speed,
                            frames_before=args.frames_before,
                            frames_after=args.frames_after,
                            pause_frames=args.pause_frames,
                            output_dir=out,
                        )
                        batch_ok += n_ok
                        batch_skip += n_skip
                    finally:
                        reader.close()
                last_idx = max(i for i, _ in pending)
                next_idx = last_idx + 1
                _save_cursor_state(state_path, next_idx)
                print(
                    f"[bounce_clips_from_hls] batch: ok={batch_ok} skipped={batch_skip}; "
                    f"cursor -> next_row_index={next_idx} ({state_path})",
                    flush=True,
                )
            else:
                if not state_on_disk:
                    _save_cursor_state(state_path, len(all_rows))
                    print(
                        f"[bounce_clips_from_hls] initial pass: no rows; "
                        f"cursor -> next_row_index=0 ({state_path})",
                        flush=True,
                    )
                else:
                    print("[bounce_clips_from_hls] no new rows", flush=True)
            state_on_disk = True
            _interruptible_sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\n[bounce_clips_from_hls] stopped (KeyboardInterrupt)", flush=True)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)


if __name__ == "__main__":
    main()
