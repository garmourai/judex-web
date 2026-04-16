#!/usr/bin/env python3
"""
Extract net crossings from trajectory_selection.jsonl (stdlib only).

Single-pass flow per frame-gap segment:
1) apply xy step-based segment splits (spike and collapse+y),
2) emit net-hit candidates on collapse+y splits,
3) detect/commit net crossings with deferred after-window validation,
4) suppress same-direction crossings within a short frame interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

DEFAULT_JSONL = "/mnt/data/cv_output/correlation/trajectory_selection.jsonl"
DEFAULT_OUT_DIR = "/mnt/data/cv_output/correlation"
EPS_Y = 1e-4
SPIKE_MIN_DISTANCE_M = 2.0

Point3D = Tuple[float, float, float]
DetectionRow = Tuple[int, Point3D]


def _parse_jsonl(path: str) -> List[DetectionRow]:
    rows: List[DetectionRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = obj.get("frame_id")
            if not isinstance(fid, int):
                continue
            sel = obj.get("current_selected_point")
            if not isinstance(sel, dict):
                continue
            try:
                x = float(sel["x"])
                y = float(sel["y"])
                z = float(sel["z"])
            except Exception:
                continue
            rows.append((fid, (x, y, z)))
    return rows


def _load_bboxes_by_frame_from_selection_jsonl(path: str) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Read bbox_by_camera from trajectory_selection.jsonl.

    Returns frame_id -> {
        bbox_source_x/y/w/h, bbox_sink_x/y/w/h
    } where each value is float or None.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}
    if not path or not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fid = obj.get("frame_id")
            sel = obj.get("current_selected_point")
            if not isinstance(fid, int) or not isinstance(sel, dict):
                continue
            bmap = sel.get("bbox_by_camera")
            if not isinstance(bmap, dict):
                continue

            def _read(cam: str, key: str) -> Optional[float]:
                bb = bmap.get(cam)
                if not isinstance(bb, dict):
                    return None
                v = bb.get(key)
                if v is None:
                    return None
                return float(v)

            out[fid] = {
                "bbox_source_x": _read("source", "x"),
                "bbox_source_y": _read("source", "y"),
                "bbox_source_w": _read("source", "w"),
                "bbox_source_h": _read("source", "h"),
                "bbox_sink_x": _read("sink", "x"),
                "bbox_sink_y": _read("sink", "y"),
                "bbox_sink_w": _read("sink", "w"),
                "bbox_sink_h": _read("sink", "h"),
            }
    return out


def _filter_rows(rows: List[DetectionRow], lo: Optional[int], hi: Optional[int]) -> List[DetectionRow]:
    out: List[DetectionRow] = []
    for fid, pt in rows:
        if lo is not None and fid < lo:
            continue
        if hi is not None and fid > hi:
            continue
        out.append((fid, pt))
    return out


def _segment_by_frame_gap(rows: List[DetectionRow], max_gap: int) -> List[List[int]]:
    if max_gap < 0:
        raise ValueError("segment_frame_gap must be >= 0")
    out: List[List[int]] = []
    cur: List[int] = []
    prev_fid: Optional[int] = None
    for fid, _ in rows:
        if prev_fid is None or (fid - prev_fid) <= max_gap:
            cur.append(fid)
        else:
            if cur:
                out.append(cur)
            cur = [fid]
        prev_fid = fid
    if cur:
        out.append(cur)
    return out


def _dist_xy(detected: Dict[int, Point3D], a: int, b: int) -> float:
    xa, ya, _ = detected[a]
    xb, yb, _ = detected[b]
    return math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)


def _side_sign(y: float, net_y: float) -> int:
    return -1 if y < net_y else 1


def _side_label(y: float, net_y: float) -> str:
    return "source_side" if y < net_y else "sink_side"


def _pick_first_side_frames(
    frames_scan_order: List[int],
    side: Dict[int, int],
    target_side: int,
    pick_count: int,
) -> Optional[List[int]]:
    out: List[int] = []
    for f in frames_scan_order:
        if side[f] == target_side:
            out.append(f)
            if len(out) >= pick_count:
                return out
    return None


def _build_crossing_event(
    frames: List[int],
    side: Dict[int, int],
    detected: Dict[int, Point3D],
    net_y: float,
    k: int,
    window_frames: int,
    pick_count: int,
) -> Optional[Dict[str, object]]:
    w = window_frames
    pc = pick_count
    if k < w:
        return None
    if k + w - 1 >= len(frames):
        return None

    old_side = side[frames[k - 1]]
    new_side = side[frames[k]]
    if old_side == new_side:
        return None

    before_scan = [frames[k - 1 - j] for j in range(w)]
    after_scan = [frames[k + j] for j in range(w)]
    frames_before = _pick_first_side_frames(before_scan, side, old_side, pc)
    frames_after = _pick_first_side_frames(after_scan, side, new_side, pc)
    if frames_before is None or frames_after is None:
        return None

    ys_b = [detected[f][1] for f in frames_before]
    ys_a = [detected[f][1] for f in frames_after]
    mean_b = sum(ys_b) / pc
    mean_a = sum(ys_a) / pc
    delta_y = mean_a - mean_b
    if delta_y > EPS_Y:
        direction = "left_to_right"
    elif delta_y < -EPS_Y:
        direction = "right_to_left"
    else:
        direction = "unknown"

    return {
        "frame": frames[k],
        "direction": direction,
        "delta_y": delta_y,
        "mean_y_before": mean_b,
        "mean_y_after": mean_a,
        "from_side": _side_label(mean_b, net_y),
        "to_side": _side_label(mean_a, net_y),
        "frames_used_before": frames_before,
        "frames_used_after": frames_after,
        "search_window_before": before_scan,
        "search_window_after": after_scan,
    }


MIN_RUN_LEN = 6   # 3 before + 1 bounce + 2 after

# Outbound run (post-crossing): close when at least REVERSE_WINDOW_MIN_MATCH steps
# in the latest REVERSE_WINDOW_SIZE frames move in reverse net-crossing y direction.
REVERSE_WINDOW_SIZE = 7
REVERSE_WINDOW_MIN_MATCH = 5
# |dy| below this (meters) is neutral: does not set crossing sign, count streak, or advance streak.
OUTBOUND_DY_NEUTRAL_ABS = 0.1


def _net_crossing_y_sign_from_event(crossing_ev: Dict[str, object]) -> Optional[int]:
    """
    Sign of net crossing in y from event direction:
    left_to_right -> +1, right_to_left -> -1. Unknown -> None (filled from first non-neutral dy).
    """
    dr = str(crossing_ev.get("direction", ""))
    if dr == "left_to_right":
        return 1
    if dr == "right_to_left":
        return -1
    return None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# Fallback when no valley bounce: crossing height vs lowest z in run (meters).
GROUND_HIT_CROSSING_Z_MIN = 0.8   # z at crossing frame must be strictly above this
GROUND_HIT_MIN_Z_MAX = 0.5       # minimum z in run must be at or below this (near ground)

# Normalize cumulative z path lengths (meters) into [0, 1] for scoring.
PATH_REF_M = 1.5
# Match valley gates: only count z steps within up to 4 frames before / after candidate.
PATH_SCORE_WINDOW = 4


def _cumulative_z_drop_before(
    run_frames: List[int],
    idx: int,
    detected: Dict[int, Point3D],
) -> float:
    """Sum of downward z steps in the last PATH_SCORE_WINDOW frames before candidate, through candidate."""
    before_z = [detected[b][2] for b in run_frames[:idx]]
    if not before_z:
        return 0.0
    pz = detected[run_frames[idx]][2]
    zs = before_z[-PATH_SCORE_WINDOW:] + [pz]
    if len(zs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(zs)):
        if zs[i - 1] > zs[i]:
            total += zs[i - 1] - zs[i]
    return total


def _cumulative_z_rise_after(
    run_frames: List[int],
    idx: int,
    detected: Dict[int, Point3D],
) -> float:
    """Sum of upward z steps from candidate through the next PATH_SCORE_WINDOW frames after."""
    after_z = [detected[a][2] for a in run_frames[idx + 1 :]]
    pz = detected[run_frames[idx]][2]
    zs = [pz] + after_z[:PATH_SCORE_WINDOW]
    if len(zs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(zs)):
        if zs[i] > zs[i - 1]:
            total += zs[i] - zs[i - 1]
    return total


def _candidate_score(
    run_frames: List[int],
    idx: int,
    detected: Dict[int, Point3D],
) -> Optional[Dict[str, float]]:
    before_frames = run_frames[:idx]
    after_frames = run_frames[idx + 1:]
    if len(before_frames) < 3 or len(after_frames) < 2:
        return None

    f = run_frames[idx]
    _, _, pz = detected[f]
    before_z = [detected[b][2] for b in before_frames]
    after_z = [detected[a][2] for a in after_frames]
    mean_before = sum(before_z) / len(before_z)
    mean_after = sum(after_z) / len(after_z)

    # Valley gate: in up-to-4 windows, candidate must be lower than
    # at least 3 points before and at least 2 points after.
    before_check = before_z[-4:]
    after_check = after_z[:4]
    if not before_check or not after_check:
        return None
    if sum(1 for zb in before_check if zb > pz) < 3:
        return None
    if sum(1 for za in after_check if za > pz) < 2:
        return None

    valley_depth = ((mean_before - pz) + (mean_after - pz)) / 2.0
    shape = _clamp01(valley_depth / 0.35)

    drop_m = _cumulative_z_drop_before(run_frames, idx, detected)
    rise_m = _cumulative_z_rise_after(run_frames, idx, detected)
    drop_n = _clamp01(drop_m / PATH_REF_M)
    rise_n = _clamp01(rise_m / PATH_REF_M)
    path_score = 0.5 * drop_n + 0.5 * rise_n

    # Combined: valley shape + local path (4 frames before/after candidate, same as gates)
    score = 0.5 * float(shape) + 0.5 * float(path_score)
    return {
        "score": score,
        "shape": float(shape),
        "path_score": float(path_score),
        "path_drop_m": float(drop_m),
        "path_rise_m": float(rise_m),
        "trend": float(path_score),
        "evidence": 0.0,
        "continuity": 1.0,
        "evidence_points_before": float(len(before_frames)),
        "evidence_points_after": float(len(after_frames)),
        "missing_nearby": 0.0,
    }


def _fallback_ground_hit_bounce(
    run_frames: List[int],
    detected: Dict[int, Point3D],
    net_y: float,
) -> Optional[Dict[str, object]]:
    """
    If no valley candidate: use lowest-z frame as ground hit when crossing is high
    and the run reaches near ground by segment end.
    """
    if len(run_frames) < MIN_RUN_LEN:
        return None
    _, _, z_cross = detected[run_frames[0]]
    if z_cross <= GROUND_HIT_CROSSING_Z_MIN:
        return None
    min_z = min(detected[f][2] for f in run_frames)
    if min_z > GROUND_HIT_MIN_Z_MAX:
        return None
    bounce_f = min(run_frames, key=lambda f: detected[f][2])
    x, y, z = detected[bounce_f]
    return {
        "bounce_frame": bounce_f,
        "x": x,
        "y": y,
        "z": z,
        "side": _side_label(y, net_y),
        "score": 0.5,
        "shape": 0.0,
        "trend": 0.0,
        "evidence": 0.0,
        "continuity": 1.0,
        "evidence_points_before": 0,
        "evidence_points_after": 0,
        "missing_nearby": 0,
        "path_drop_m": 0.0,
        "path_rise_m": 0.0,
        "path_score": 0.0,
        "run_start": run_frames[0],
        "run_end": run_frames[-1],
        "run_length": len(run_frames),
        "bounce_mode": "ground_hit_fallback",
    }


def _score_run_best_bounce(
    run_frames: List[int],
    detected: Dict[int, Point3D],
    net_y: float,
) -> Optional[Dict[str, object]]:
    if len(run_frames) < MIN_RUN_LEN:
        return None
    best: Optional[Tuple[int, Dict[str, float]]] = None
    for idx in range(3, len(run_frames) - 2):
        metrics = _candidate_score(run_frames, idx, detected)
        if metrics is None:
            continue
        if best is None or metrics["score"] > best[1]["score"]:
            best = (run_frames[idx], metrics)
    if best is not None:
        frame, m = best
        x, y, z = detected[frame]
        out: Dict[str, object] = {
            "bounce_frame": frame,
            "x": x,
            "y": y,
            "z": z,
            "side": _side_label(y, net_y),
            "score": m["score"],
            "shape": m["shape"],
            "trend": m["trend"],
            "evidence": m["evidence"],
            "continuity": m["continuity"],
            "evidence_points_before": int(m["evidence_points_before"]),
            "evidence_points_after": int(m["evidence_points_after"]),
            "missing_nearby": int(m["missing_nearby"]),
            "path_drop_m": float(m.get("path_drop_m", 0.0)),
            "path_rise_m": float(m.get("path_rise_m", 0.0)),
            "path_score": float(m.get("path_score", m["trend"])),
            "run_start": run_frames[0],
            "run_end": run_frames[-1],
            "run_length": len(run_frames),
            "bounce_mode": "valley",
        }
        return out
    return _fallback_ground_hit_bounce(run_frames, detected, net_y)


def _has_min_net_low_height_points(
    seg_frames: List[int],
    detected: Dict[int, Point3D],
    start_idx: int,
    net_y: float,
    y_band: float,
    max_height: float,
    min_points: int,
) -> bool:
    """
    True when there are at least min_points consecutive detections from start_idx
    that stay within net_y +/- y_band and have z < max_height.
    """
    if start_idx < 0 or start_idx >= len(seg_frames):
        return False
    needed = max(1, min_points)
    if start_idx + needed - 1 >= len(seg_frames):
        return False
    cnt = 0
    for j in range(start_idx, len(seg_frames)):
        f = seg_frames[j]
        _, y, z = detected[f]
        if abs(y - net_y) <= y_band and z < max_height:
            cnt += 1
            if cnt >= needed:
                return True
        else:
            break
    return False


def _process_segment_single_pass(
    seg_frames: List[int],
    detected: Dict[int, Point3D],
    net_y: float,
    window_frames: int,
    pick_count: int,
    step_ma_window: int,
    step_max_ratio: float,
    step_split_y_band: float,
    step_split_max_height: float,
    step_split_min_points: int,
    crossing_suppress_gap: int,
    segment_base_index: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int, List[Dict[str, object]]]:
    """
    Returns (crossings, net_hits, final_subsegment_count, bounces).
    """
    if not seg_frames:
        return [], [], 0, []

    w = max(2, window_frames)
    pc = max(2, pick_count)
    ma_w = max(3, min(5, step_ma_window))

    crossings: List[Dict[str, object]] = []
    net_hits: List[Dict[str, object]] = []
    bounces: List[Dict[str, object]] = []

    last_emitted_crossing_frame: Optional[int] = None
    last_emitted_direction: Optional[str] = None

    final_subsegment_count = 0
    current_subseg_idx = 0
    subseg_start_frame = seg_frames[0]

    cur_frames: List[int] = [seg_frames[0]]
    side: Dict[int, int] = {seg_frames[0]: _side_sign(detected[seg_frames[0]][1], net_y)}
    pending_k: Deque[int] = deque()
    d_hist: Deque[float] = deque()
    d_sum = 0.0

    # -- Outbound run state for inline bounce detection --
    run_frames: List[int] = []
    run_crossing: Optional[Dict[str, object]] = None
    # +1 / -1: net crossing direction in y; if None (unknown crossing), set from first non-neutral dy
    net_crossing_y_sign: Optional[int] = None
    reverse_window_flags: Deque[int] = deque(maxlen=REVERSE_WINDOW_SIZE)

    def close_active_run() -> None:
        nonlocal run_frames, run_crossing, reverse_window_flags, net_crossing_y_sign
        if run_crossing is not None and len(run_frames) >= MIN_RUN_LEN:
            bounce = _score_run_best_bounce(run_frames, detected, net_y)
            if bounce is not None:
                bounce["crossing_frame"] = run_crossing["frame"]
                bounce["crossing_direction"] = run_crossing["direction"]
                bounce["segment_base_index"] = segment_base_index
                bounce["subsegment_index"] = run_crossing.get("subsegment_index", current_subseg_idx)
                bounces.append(bounce)
        run_frames = []
        run_crossing = None
        reverse_window_flags.clear()
        net_crossing_y_sign = None

    def append_outbound_frame(f: int) -> None:
        nonlocal run_frames, reverse_window_flags, net_crossing_y_sign
        if run_crossing is None:
            return
        y = detected[f][1]

        if not run_frames:
            run_frames.append(f)
            return

        prev_y = detected[run_frames[-1]][1]
        dy = y - prev_y

        if abs(dy) < OUTBOUND_DY_NEUTRAL_ABS:
            step_sign = 0  # neutral: noise / outlier
        elif dy > 0:
            step_sign = 1
        else:
            step_sign = -1

        run_frames.append(f)

        if net_crossing_y_sign is None and step_sign != 0:
            net_crossing_y_sign = step_sign  # unknown crossing: infer crossing y-sign from first non-neutral dy

        # Streak: dy aligned with *reverse* of net crossing y direction (i.e. against crossing sense in y)
        reverse_net_crossing_y_sign: Optional[int] = (
            -net_crossing_y_sign if net_crossing_y_sign is not None else None
        )

        matches_reverse_of_crossing = (
            step_sign != 0
            and reverse_net_crossing_y_sign is not None
            and step_sign == reverse_net_crossing_y_sign
        )
        reverse_window_flags.append(1 if matches_reverse_of_crossing else 0)

        if (
            len(reverse_window_flags) == REVERSE_WINDOW_SIZE
            and sum(reverse_window_flags) >= REVERSE_WINDOW_MIN_MATCH
        ):
            del run_frames[-REVERSE_WINDOW_SIZE:]
            reverse_window_flags.clear()
            net_crossing_y_sign = None
            close_active_run()

    def start_run_from_crossing(crossing_ev: Dict[str, object], seed_frames: List[int]) -> None:
        nonlocal run_frames, run_crossing, reverse_window_flags, net_crossing_y_sign
        run_crossing = crossing_ev
        run_frames = []
        reverse_window_flags.clear()
        net_crossing_y_sign = _net_crossing_y_sign_from_event(crossing_ev)
        for f in seed_frames:
            append_outbound_frame(f)

    def track_outbound_frame(f: int) -> None:
        append_outbound_frame(f)

    def emit_ready_crossings() -> None:
        nonlocal last_emitted_crossing_frame, last_emitted_direction
        last_idx = len(cur_frames) - 1
        while pending_k and (pending_k[0] + w - 1) <= last_idx:
            k = pending_k.popleft()
            ev = _build_crossing_event(
                cur_frames, side, detected, net_y, k, w, pc
            )
            if ev is None:
                continue
            fr = int(ev["frame"])  # type: ignore[index]
            dr = str(ev["direction"])  # type: ignore[index]
            if (
                last_emitted_crossing_frame is not None
                and last_emitted_direction is not None
                and dr == last_emitted_direction
                and (fr - last_emitted_crossing_frame) <= crossing_suppress_gap
            ):
                continue
            ev["segment_base_index"] = segment_base_index
            ev["subsegment_index"] = current_subseg_idx
            ev["subseg_start_frame"] = subseg_start_frame
            crossings.append(ev)
            last_emitted_crossing_frame = fr
            last_emitted_direction = dr

    def reset_subsegment(start_frame: int) -> None:
        nonlocal cur_frames, side, pending_k, d_hist, d_sum, final_subsegment_count
        nonlocal current_subseg_idx, subseg_start_frame
        final_subsegment_count += 1
        current_subseg_idx += 1
        subseg_start_frame = start_frame
        cur_frames = [start_frame]
        side = {start_frame: _side_sign(detected[start_frame][1], net_y)}
        pending_k = deque()
        d_hist = deque()
        d_sum = 0.0

    for i in range(1, len(seg_frames)):
        prev_f = seg_frames[i - 1]
        cur_f = seg_frames[i]
        frame_delta = max(1, cur_f - prev_f)
        d_raw = _dist_xy(detected, prev_f, cur_f)
        d = d_raw / float(frame_delta)

        split_spike = False
        split_collapse = False
        if len(d_hist) >= 3:
            ma = d_sum / float(len(d_hist))
            split_spike = d > (step_max_ratio * ma) and d > SPIKE_MIN_DISTANCE_M
            split_collapse = _has_min_net_low_height_points(
                seg_frames=seg_frames,
                detected=detected,
                start_idx=i,
                net_y=net_y,
                y_band=step_split_y_band,
                max_height=step_split_max_height,
                min_points=step_split_min_points,
            )

        if split_spike or split_collapse:
            close_active_run()
            if split_collapse:
                net_hits.append(
                    {
                        "frame": cur_f,
                        "segment_base_index": segment_base_index,
                        "subsegment_index": current_subseg_idx,
                        "reason": "collapse_plus_y",
                        "step_distance_xy": d,
                        "step_distance_xy_raw": d_raw,
                        "frame_delta": frame_delta,
                        "y": detected[cur_f][1],
                        "z": detected[cur_f][2],
                        "net_y": net_y,
                        "y_distance_to_net": abs(detected[cur_f][1] - net_y),
                        "max_height_threshold": step_split_max_height,
                        "min_points_threshold": step_split_min_points,
                        "jitter_confirmed": None,
                    }
                )
            reset_subsegment(cur_f)
            continue

        cur_frames.append(cur_f)
        side[cur_f] = _side_sign(detected[cur_f][1], net_y)
        if side[cur_frames[-2]] != side[cur_frames[-1]]:
            pending_k.append(len(cur_frames) - 1)

        d_hist.append(d)
        d_sum += d
        if len(d_hist) > ma_w:
            d_sum -= d_hist.popleft()

        crossings_before = len(crossings)
        emit_ready_crossings()

        if len(crossings) > crossings_before:
            close_active_run()
            latest = crossings[-1]
            cf = int(latest["frame"])  # type: ignore[arg-type]
            seed_start = cur_frames.index(cf)
            start_run_from_crossing(latest, cur_frames[seed_start:])
        elif run_crossing is not None:
            track_outbound_frame(cur_f)

    emit_ready_crossings()
    close_active_run()
    final_subsegment_count += 1
    return crossings, net_hits, final_subsegment_count, bounces


def main() -> None:
    p = argparse.ArgumentParser(description="Net crossings from trajectory_selection.jsonl")
    p.add_argument("--trajectory-jsonl", default=DEFAULT_JSONL)
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--net-y", type=float, default=6.7)

    p.add_argument(
        "--segment-frame-gap",
        type=int,
        default=5,
        help="Start new base segment when next frame_id - prev frame_id > this (default: 5)",
    )
    p.add_argument(
        "--crossing-suppress-gap",
        type=int,
        default=5,
        help="Suppress same-direction crossing if within this many frame ids from last emitted (default: 5)",
    )
    p.add_argument(
        "--step-distance-ma-window",
        type=int,
        default=4,
        help="MA window for xy step distance (clamped to 3..5, default: 4)",
    )
    p.add_argument(
        "--step-distance-max-ratio",
        type=float,
        default=2.0,
        help="Spike split when d_xy > ratio * MA_xy (default: 2.0)",
    )
    p.add_argument(
        "--step-split-max-height",
        type=float,
        default=0.5,
        help="Collapse split requires z < this (default: 0.5)",
    )
    p.add_argument(
        "--step-split-min-points",
        type=int,
        default=5,
        help="Collapse split requires at least this many consecutive points in net band + low height (default: 5)",
    )
    p.add_argument(
        "--step-split-y-band",
        type=float,
        default=0.5,
        help="Collapse split requires abs(y-net_y) <= this (default: 0.5)",
    )

    p.add_argument(
        "--window-frames",
        type=int,
        default=4,
        help="Max detections to scan before/after crossing (default: 4)",
    )
    p.add_argument(
        "--min-hits-in-window",
        type=int,
        default=2,
        help="Take first N detections on each side within each scan (default: 2)",
    )

    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    args = p.parse_args()

    rows = _parse_jsonl(args.trajectory_jsonl)
    if not rows:
        raise SystemExit("No valid frame_id/current_selected_point lines in JSONL")

    rows = _filter_rows(rows, args.start_frame, args.end_frame)
    if not rows:
        raise SystemExit("No valid rows after start/end frame filtering")

    detected: Dict[int, Point3D] = {fid: pt for fid, pt in rows}
    frame_ids = [fid for fid, _ in rows]

    base_segments = _segment_by_frame_gap(rows, max(0, args.segment_frame_gap))

    win = max(2, args.window_frames)
    pick_n = max(2, args.min_hits_in_window)
    ma_window = max(3, min(5, args.step_distance_ma_window))

    all_crossings: List[Dict[str, object]] = []
    all_net_hits: List[Dict[str, object]] = []
    all_bounces: List[Dict[str, object]] = []
    final_segment_count = 0

    for bi, seg_frames in enumerate(base_segments):
        c, h, sub_count, b = _process_segment_single_pass(
            seg_frames=seg_frames,
            detected=detected,
            net_y=args.net_y,
            window_frames=win,
            pick_count=pick_n,
            step_ma_window=ma_window,
            step_max_ratio=max(1e-9, args.step_distance_max_ratio),
            step_split_y_band=max(0.0, args.step_split_y_band),
            step_split_max_height=args.step_split_max_height,
            step_split_min_points=max(1, args.step_split_min_points),
            crossing_suppress_gap=max(0, args.crossing_suppress_gap),
            segment_base_index=bi,
        )
        all_crossings.extend(c)
        all_net_hits.extend(h)
        all_bounces.extend(b)
        final_segment_count += sub_count

    # Enrich bounce rows with per-camera bbox coordinates from trajectory_selection.jsonl.
    bbox_by_frame = _load_bboxes_by_frame_from_selection_jsonl(args.trajectory_jsonl)
    for b in all_bounces:
        bf_obj = b.get("bounce_frame", -1)
        bf = int(bf_obj) if isinstance(bf_obj, int) else -1
        m = bbox_by_frame.get(bf, {})
        b["bbox_source_x"] = m.get("bbox_source_x")
        b["bbox_source_y"] = m.get("bbox_source_y")
        b["bbox_source_w"] = m.get("bbox_source_w")
        b["bbox_source_h"] = m.get("bbox_source_h")
        b["bbox_sink_x"] = m.get("bbox_sink_x")
        b["bbox_sink_y"] = m.get("bbox_sink_y")
        b["bbox_sink_w"] = m.get("bbox_sink_w")
        b["bbox_sink_h"] = m.get("bbox_sink_h")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "net_crossings.json")
    csv_path = os.path.join(args.output_dir, "net_crossings.csv")
    bounce_csv_path = os.path.join(args.output_dir, "bounce_events.csv")

    payload = {
        "input_jsonl": args.trajectory_jsonl,
        "frame_range": [frame_ids[0], frame_ids[-1]],
        "detected_frame_count": len(rows),
        "base_segment_count": len(base_segments),
        "segment_count": final_segment_count,
        "segment_frame_gap": args.segment_frame_gap,
        "crossing_suppress_gap": args.crossing_suppress_gap,
        "step_distance_ma_window": ma_window,
        "step_distance_max_ratio": args.step_distance_max_ratio,
        "step_split_y_band_m": args.step_split_y_band,
        "step_split_max_height_m": args.step_split_max_height,
        "step_split_min_points": args.step_split_min_points,
        "step_distance_xy_only": True,
        "window_frames": win,
        "points_per_side": pick_n,
        "min_hits_in_window": pick_n,
        "net_y": args.net_y,
        "direction_note": "left_to_right means positive delta_y (mean y after > mean y before)",
        "crossing_count": len(all_crossings),
        "net_hit_count": len(all_net_hits),
        "bounce_count": len(all_bounces),
        "crossings": all_crossings,
        "net_hits": all_net_hits,
        "bounces": all_bounces,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fields = [
        "segment_base_index",
        "frame",
        "direction",
        "delta_y",
        "mean_y_before",
        "mean_y_after",
        "from_side",
        "to_side",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_crossings:
            writer.writerow({k: row[k] for k in fields})

    bounce_fields = [
        "segment_base_index",
        "crossing_frame",
        "crossing_direction",
        "bounce_frame",
        "x",
        "y",
        "z",
        "side",
        "score",
        "shape",
        "path_score",
        "path_drop_m",
        "path_rise_m",
        "trend",
        "evidence",
        "continuity",
        "evidence_points_before",
        "evidence_points_after",
        "missing_nearby",
        "run_start",
        "run_end",
        "run_length",
        "bbox_source_x",
        "bbox_source_y",
        "bbox_source_w",
        "bbox_source_h",
        "bbox_sink_x",
        "bbox_sink_y",
        "bbox_sink_w",
        "bbox_sink_h",
    ]
    with open(bounce_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bounce_fields)
        writer.writeheader()
        for row in all_bounces:
            writer.writerow({k: row[k] for k in bounce_fields})

    # -- Unified summary CSV: one row per crossing with bounce info --
    unified_csv_path = os.path.join(args.output_dir, "net_crossings_summary.csv")
    bounce_by_crossing = {int(b["crossing_frame"]): b for b in all_bounces}  # type: ignore[arg-type]

    seg_frame_ranges: Dict[int, Tuple[int, int]] = {}
    for bi, sf in enumerate(base_segments):
        seg_frame_ranges[bi] = (sf[0], sf[-1])

    unified_fields = [
        "segment",
        "seg_start_frame",
        "seg_end_frame",
        "subsegment",
        "subseg_start_frame",
        "event_type",
        "crossing_frame",
        "crossing_direction",
        "from_side",
        "to_side",
        "delta_y",
        "bounce_frame",
        "bounce_x",
        "bounce_y",
        "bounce_z",
        "bounce_side",
        "bounce_score",
        "bounce_shape",
        "bounce_trend",
        "bounce_evidence",
        "bounce_continuity",
        "run_start",
        "run_end",
        "run_length",
    ]
    unified_rows: List[Dict[str, object]] = []
    for c in all_crossings:
        sbi = int(c["segment_base_index"])  # type: ignore[arg-type]
        seg_s, seg_e = seg_frame_ranges.get(sbi, (0, 0))
        cf = int(c["frame"])  # type: ignore[arg-type]
        b = bounce_by_crossing.get(cf)
        row: Dict[str, object] = {
            "segment": sbi,
            "seg_start_frame": seg_s,
            "seg_end_frame": seg_e,
            "subsegment": c.get("subsegment_index", ""),
            "subseg_start_frame": c.get("subseg_start_frame", ""),
            "event_type": "crossing",
            "crossing_frame": cf,
            "crossing_direction": c["direction"],
            "from_side": c["from_side"],
            "to_side": c["to_side"],
            "delta_y": c["delta_y"],
        }
        if b:
            row["bounce_frame"] = b["bounce_frame"]
            row["bounce_x"] = b["x"]
            row["bounce_y"] = b["y"]
            row["bounce_z"] = b["z"]
            row["bounce_side"] = b["side"]
            row["bounce_score"] = b["score"]
            row["bounce_shape"] = b["shape"]
            row["bounce_trend"] = b["trend"]
            row["bounce_evidence"] = b["evidence"]
            row["bounce_continuity"] = b["continuity"]
            row["run_start"] = b["run_start"]
            row["run_end"] = b["run_end"]
            row["run_length"] = b["run_length"]
        unified_rows.append(row)

    for nh in all_net_hits:
        sbi = int(nh["segment_base_index"])  # type: ignore[arg-type]
        seg_s, seg_e = seg_frame_ranges.get(sbi, (0, 0))
        unified_rows.append({
            "segment": sbi,
            "seg_start_frame": seg_s,
            "seg_end_frame": seg_e,
            "subsegment": nh.get("subsegment_index", ""),
            "event_type": "net_hit",
            "crossing_frame": nh["frame"],
            "from_side": "",
            "to_side": "",
        })

    unified_rows.sort(key=lambda r: (int(r.get("segment", 0)), int(r.get("crossing_frame", 0))))

    with open(unified_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=unified_fields)
        writer.writeheader()
        for row in unified_rows:
            writer.writerow(row)

    print(
        f"[net_crossings_from_jsonl] base_segments={len(base_segments)} "
        f"final_segments={final_segment_count} crossings={len(all_crossings)} "
        f"net_hits={len(all_net_hits)} bounces={len(all_bounces)}"
    )
    print(f"[net_crossings_from_jsonl] wrote {json_path}")
    print(f"[net_crossings_from_jsonl] wrote {csv_path}")
    print(f"[net_crossings_from_jsonl] wrote {bounce_csv_path}")
    print(f"[net_crossings_from_jsonl] wrote {unified_csv_path}")


if __name__ == "__main__":
    main()
