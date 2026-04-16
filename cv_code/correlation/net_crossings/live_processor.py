"""
Live net-crossing extraction using in-memory trajectory points (same logic as
net_crossings_from_jsonl.main, without subprocess / without re-parsing JSONL for x,y,z).

Bbox enrichment still uses trajectory_selection.jsonl via
_load_bboxes_by_frame_from_selection_jsonl (file must exist after JSONL append).
"""

from __future__ import annotations

import csv
import json
import os
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

# Loaded lazily — import cv_code/__init__.py pulls torch; load by file path instead.
_nc = None


def _net_crossings_module():
    global _nc
    if _nc is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        path = os.path.join(repo_root, "cv_code", "net_crossings_from_jsonl.py")
        spec = importlib.util.spec_from_file_location("net_crossings_from_jsonl_impl", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _nc = mod
    return _nc


DetectionRow = Tuple[int, Tuple[float, float, float]]  # frame_id, (x,y,z)


class LiveNetCrossingsAccumulator:
    """Accumulates (frame -> selected 3D point) across correlation batches."""

    def __init__(self) -> None:
        self._detected: Dict[int, Tuple[float, float, float]] = {}

    def ingest_frame_decisions(self, frame_decisions: List[Dict[str, Any]]) -> None:
        for rec in frame_decisions:
            fid = int(rec["frame"])
            sel = next((c for c in rec.get("active_candidates", []) if c.get("is_selected")), None)
            if sel is None:
                continue
            self._detected[fid] = (
                float(sel["x"]),
                float(sel["y"]),
                float(sel["z"]),
            )

    def rows(self) -> List[DetectionRow]:
        return sorted(self._detected.items(), key=lambda t: t[0])

    def clear(self) -> None:
        self._detected.clear()


def run_net_crossings_from_accumulator(
    accumulator: LiveNetCrossingsAccumulator,
    correlation_output_dir: str,
    bbox_jsonl_path: str,
) -> bool:
    """
    Run full net-crossing pipeline on all accumulated points and write outputs
    (overwrite with complete state — correct for live growing trajectories).
    """
    nc = _net_crossings_module()
    rows: List[DetectionRow] = accumulator.rows()
    if not rows:
        return True

    detected: Dict[int, Any] = {fid: pt for fid, pt in rows}
    frame_ids = [fid for fid, _ in rows]

    segment_frame_gap = 5
    crossing_suppress_gap = 5
    step_distance_ma_window = 4
    step_distance_max_ratio = 2.0
    step_split_max_height = 0.5
    step_split_min_points = 5
    step_split_y_band = 0.5
    window_frames = 4
    min_hits_in_window = 2
    net_y = 6.7

    base_segments = nc._segment_by_frame_gap(rows, max(0, segment_frame_gap))

    win = max(2, window_frames)
    pick_n = max(2, min_hits_in_window)
    ma_window = max(3, min(5, step_distance_ma_window))

    all_crossings: List[Dict[str, object]] = []
    all_net_hits: List[Dict[str, object]] = []
    all_bounces: List[Dict[str, object]] = []
    final_segment_count = 0

    for bi, seg_frames in enumerate(base_segments):
        c, h, sub_count, b = nc._process_segment_single_pass(
            seg_frames=seg_frames,
            detected=detected,
            net_y=net_y,
            window_frames=win,
            pick_count=pick_n,
            step_ma_window=ma_window,
            step_max_ratio=max(1e-9, step_distance_max_ratio),
            step_split_y_band=max(0.0, step_split_y_band),
            step_split_max_height=step_split_max_height,
            step_split_min_points=max(1, step_split_min_points),
            crossing_suppress_gap=max(0, crossing_suppress_gap),
            segment_base_index=bi,
        )
        all_crossings.extend(c)
        all_net_hits.extend(h)
        all_bounces.extend(b)
        final_segment_count += sub_count

    bbox_by_frame = {}
    if bbox_jsonl_path and os.path.exists(bbox_jsonl_path):
        bbox_by_frame = nc._load_bboxes_by_frame_from_selection_jsonl(bbox_jsonl_path)
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

    os.makedirs(correlation_output_dir, exist_ok=True)
    json_path = os.path.join(correlation_output_dir, "net_crossings.json")
    csv_path = os.path.join(correlation_output_dir, "net_crossings.csv")
    bounce_csv_path = os.path.join(correlation_output_dir, "bounce_events.csv")

    payload = {
        "input_mode": "live_in_memory_rows",
        "input_jsonl_bbox_source": bbox_jsonl_path,
        "frame_range": [frame_ids[0], frame_ids[-1]],
        "detected_frame_count": len(rows),
        "base_segment_count": len(base_segments),
        "segment_count": final_segment_count,
        "segment_frame_gap": segment_frame_gap,
        "crossing_suppress_gap": crossing_suppress_gap,
        "step_distance_ma_window": ma_window,
        "step_distance_max_ratio": step_distance_max_ratio,
        "step_split_y_band_m": step_split_y_band,
        "step_split_max_height_m": step_split_max_height,
        "step_split_min_points": step_split_min_points,
        "step_distance_xy_only": True,
        "window_frames": win,
        "points_per_side": pick_n,
        "min_hits_in_window": pick_n,
        "net_y": net_y,
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

    unified_csv_path = os.path.join(correlation_output_dir, "net_crossings_summary.csv")
    bounce_by_crossing = {int(b["crossing_frame"]): b for b in all_bounces}

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
        sbi = int(c["segment_base_index"])
        seg_s, seg_e = seg_frame_ranges.get(sbi, (0, 0))
        cf = int(c["frame"])
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
        sbi = int(nh["segment_base_index"])
        seg_s, seg_e = seg_frame_ranges.get(sbi, (0, 0))
        unified_rows.append(
            {
                "segment": sbi,
                "seg_start_frame": seg_s,
                "seg_end_frame": seg_e,
                "subsegment": nh.get("subsegment_index", ""),
                "event_type": "net_hit",
                "crossing_frame": nh["frame"],
                "from_side": "",
                "to_side": "",
            }
        )

    unified_rows.sort(key=lambda r: (int(r.get("segment", 0)), int(r.get("crossing_frame", 0))))

    with open(unified_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=unified_fields)
        writer.writeheader()
        for row in unified_rows:
            writer.writerow(row)

    print(
        f"[net_crossings live] base_segments={len(base_segments)} "
        f"final_segments={final_segment_count} crossings={len(all_crossings)} "
        f"net_hits={len(all_net_hits)} bounces={len(all_bounces)} "
        f"frames=[{frame_ids[0]}, {frame_ids[-1]}]"
    )
    return True
