#!/usr/bin/env python3
"""
Create a 3D Plotly HTML animation from trajectory_selection.jsonl.

Rules implemented:
- Read JSONL line-by-line.
- Keep only valid JSON objects with integer frame_id.
- Preserve file order (assumed incremental frame_id).
- Use only current_selected_point + selected_trajectory_id (no fallbacks).
- Plot one selected 3D point per frame (no trajectory/trail model).
- Every integer frame from min(frame_id) to max(frame_id) is included (gaps where JSONL omits
  no-detection frames are filled; those steps show an empty trail).
- Draw court from a world-points TXT (``name x y z`` or ``x y z`` per line): all points as
  markers; outer boundary = convex hull of near-ground points (or AABB if scipy unavailable).
- Display frame number in title and fixed annotation.
- 3D scene uses a display frame: origin (0,0,0) at the court corner with minimum world X and Y
  (top-left in view), +X to the right, +Y downward, +Z upward; marker hovers still show world XYZ.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_CV_OUTPUT_DIR = "/mnt/data/cv_output"
DEFAULT_JSONL_PATH = os.path.join(
    DEFAULT_CV_OUTPUT_DIR, "correlation", "trajectory_selection.jsonl"
)
DEFAULT_HTML_OUTPUT = os.path.join(
    DEFAULT_CV_OUTPUT_DIR, "selected_point_3d_visualization.html"
)
DEFAULT_WORLD_POINTS_PATH = (
    "/home/ubuntu/test_work/judex-web/tools/pickleball_calib/worldpickleball.txt"
)
# Fixed default z-axis extent (meters) for the 3D scene; two vertical guides span this range.
DEFAULT_SCENE_Z_RANGE = (-2.0, 10.0)

# Plot uses display axes; hovers show world coordinates (meters).
MARKER_HOVER_WORLD = (
    "world x: %{customdata[0]:.4f} m<br>world y: %{customdata[1]:.4f} m<br>world z: %{customdata[2]:.4f} m"
    "<extra></extra>"
)


def _load_world_points(world_points_file: Optional[str]) -> Optional[np.ndarray]:
    """Load world points from explicit path or fallbacks (``x y z`` or ``name x y z`` per line)."""
    candidates: List[str] = []
    if world_points_file:
        candidates.append(world_points_file)
    candidates.append(DEFAULT_WORLD_POINTS_PATH)
    candidates.append(
        os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "calibration_testing",
                "source",
                "world_points.txt",
            )
        )
    )
    candidates.append(os.path.join(os.path.dirname(__file__), "world_points_3D.txt"))

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            pts: List[Tuple[float, float, float]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").split()
                    # ``x y z`` or ``name x y z`` (last three tokens are coordinates)
                    if len(parts) < 3:
                        continue
                    nums = parts[-3:]
                    try:
                        x, y, z = float(nums[0]), float(nums[1]), float(nums[2])
                    except ValueError:
                        continue
                    pts.append((x, y, z))
            if pts:
                return np.asarray(pts, dtype=float)
        except Exception:
            continue
    return None


def _convex_hull_closed_loop(
    points_xyz: np.ndarray, z_ground_tol: float = 0.55
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Closed court boundary in the XY plane: 2D convex hull of near-ground points.
    Uses scipy if available; returns None if hull cannot be built.
    """
    if points_xyz.shape[0] < 3:
        return None
    near = points_xyz[np.abs(points_xyz[:, 2]) <= z_ground_tol]
    g = near if near.shape[0] >= 3 else points_xyz
    xy = g[:, :2].astype(np.float64)
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(xy)
        idx = hull.vertices
    except Exception:
        return None
    x_closed = np.append(g[idx, 0], g[idx[0], 0])
    y_closed = np.append(g[idx, 1], g[idx[0], 1])
    z_closed = np.append(g[idx, 2], g[idx[0], 2])
    return x_closed, y_closed, z_closed


def _parse_jsonl_selected_points(
    jsonl_path: str,
) -> Tuple[List[int], Dict[int, Tuple[int, Tuple[float, float, float]]]]:
    """
    Parse JSONL in file order and return:
      - frame_ids in input order (valid integer frame_id lines only)
      - selected point map: frame_id -> (trajectory_id, (x,y,z))
    """
    frame_ids: List[int] = []
    selected_by_frame: Dict[int, Tuple[int, Tuple[float, float, float]]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            frame_id = obj.get("frame_id")
            if not isinstance(frame_id, int):
                continue

            frame_ids.append(frame_id)

            selected = obj.get("current_selected_point")
            selected_tid = obj.get("selected_trajectory_id")
            if not isinstance(selected, dict):
                continue
            if not isinstance(selected_tid, int):
                continue
            try:
                x = float(selected["x"])
                y = float(selected["y"])
                z = float(selected["z"])
            except Exception:
                continue

            # One authoritative point per frame.
            selected_by_frame[frame_id] = (selected_tid, (x, y, z))

    return frame_ids, selected_by_frame


def create_3d_trajectory_visualization_from_jsonl(
    jsonl_path: str,
    output_file: str,
    world_points_file: Optional[str] = None,
    title: str = "3D Selected Point Visualization",
    chunk_size: int = 1000,
) -> None:
    """Generate standalone Plotly HTML from JSONL-selected per-frame points."""
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError as e:
        raise RuntimeError("plotly is required. Install with: pip install plotly") from e

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    frame_ids, selected_by_frame = _parse_jsonl_selected_points(jsonl_path)
    if not frame_ids:
        raise ValueError("No valid JSON lines with integer frame_id were found.")

    # Unique frame ids in file order (sparse: only frames that appear in JSONL at least once).
    seen = set()
    ordered_sparse: List[int] = []
    for fid in frame_ids:
        if fid in seen:
            continue
        seen.add(fid)
        ordered_sparse.append(fid)

    # Dense range: include every integer between min and max so frames with no JSONL line
    # (typical when the pipeline skips writing no-detection frames) are still animated empty.
    lo, hi = min(ordered_sparse), max(ordered_sparse)
    ordered_frames = list(range(lo, hi + 1))

    world_points = _load_world_points(world_points_file)
    boundary_hull: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    if world_points is not None:
        main_court_points = world_points
        min_x = float(np.min(world_points[:, 0]))
        max_x = float(np.max(world_points[:, 0]))
        min_y = float(np.min(world_points[:, 1]))
        max_y = float(np.max(world_points[:, 1]))
        min_z = float(np.min(world_points[:, 2]))
        max_z = float(np.max(world_points[:, 2]))
        unique_y = np.unique(np.round(world_points[:, 1], 3))
        net_candidates = unique_y[np.isclose(unique_y, 6.7, atol=0.02)] if unique_y.size else []
        net_y = float(net_candidates[0]) if np.size(net_candidates) > 0 else (min_y + max_y) / 2.0
        center_x = (min_x + max_x) / 2.0
        nvz_offset_m = 2.1336
        nvz_y_top = net_y - nvz_offset_m
        nvz_y_bottom = net_y + nvz_offset_m
        boundary_hull = _convex_hull_closed_loop(world_points)
        pad = 1.0
        # X/Y reversed to match court world frame; Z uses fixed default extent (see DEFAULT_SCENE_Z_RANGE).
        axis_limits = {
            "x": [max_x + pad, min_x - pad],
            "y": [max_y + pad, min_y - pad],
            "z": [DEFAULT_SCENE_Z_RANGE[0], DEFAULT_SCENE_Z_RANGE[1]],
        }
    else:
        main_court_points = None
        min_x, max_x = -1.0, 7.0
        min_y, max_y = -1.0, 15.0
        net_y = (min_y + max_y) / 2.0
        center_x = (min_x + max_x) / 2.0
        nvz_offset_m = 2.1336
        nvz_y_top = net_y - nvz_offset_m
        nvz_y_bottom = net_y + nvz_offset_m

    # Display frame: (0,0,0) at top-left = min world X and Y; +X right, +Y down, +Z up.
    pad = 1.0
    ox = min_x
    oy = min_y
    span_x = max_x - min_x
    span_y = max_y - min_y
    axis_limits = {
        "x": [-pad, span_x + pad],
        "y": [span_y + pad, -pad],
        "z": [DEFAULT_SCENE_Z_RANGE[0], DEFAULT_SCENE_Z_RANGE[1]],
    }

    def _court_traces(showlegend: bool):
        traces: List["go.Scatter3d"] = []
        cx_d = center_x - ox
        net_d = net_y - oy
        nvz_top_d = nvz_y_top - oy
        nvz_bot_d = nvz_y_bottom - oy
        # Anchor points force a stable global plotting cube across chunks.
        ax0, ax1 = axis_limits["x"][0], axis_limits["x"][1]
        ay0, ay1 = axis_limits["y"][0], axis_limits["y"][1]
        az0, az1 = axis_limits["z"][0], axis_limits["z"][1]
        x_min, x_max = min(ax0, ax1), max(ax0, ax1)
        y_min, y_max = min(ay0, ay1), max(ay0, ay1)
        z_min, z_max = min(az0, az1), max(az0, az1)
        traces.append(
            go.Scatter3d(
                x=[x_min, x_max],
                y=[y_min, y_max],
                z=[z_min, z_max],
                mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        z_lo, z_hi = z_min, z_max
        traces.append(
            go.Scatter3d(
                x=[0.0, 0.0],
                y=[net_d, net_d],
                z=[z_lo, z_hi],
                mode="lines",
                line=dict(color="rgba(90, 90, 90, 0.75)", width=4),
                name="Z extent (vertical)",
                showlegend=showlegend,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[span_x, span_x],
                y=[net_d, net_d],
                z=[z_lo, z_hi],
                mode="lines",
                line=dict(color="rgba(90, 90, 90, 0.75)", width=4),
                name="" if showlegend else None,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if main_court_points is not None:
            wx = main_court_points[:, 0]
            wy = main_court_points[:, 1]
            wz = main_court_points[:, 2]
            traces.append(
                go.Scatter3d(
                    x=wx - ox,
                    y=wy - oy,
                    z=wz,
                    mode="markers",
                    marker=dict(size=4, color="rgba(100, 100, 100, 0.75)"),
                    name="Court points (world txt)",
                    showlegend=showlegend,
                    customdata=np.column_stack((wx, wy, wz)),
                    hovertemplate=MARKER_HOVER_WORLD,
                )
            )
        if boundary_hull is not None:
            hx, hy, hz = boundary_hull
            traces.append(
                go.Scatter3d(
                    x=hx - ox,
                    y=hy - oy,
                    z=hz,
                    mode="lines",
                    line=dict(color="rgba(139, 0, 0, 1.0)", width=8),
                    name="Court boundary (convex hull)",
                    showlegend=showlegend,
                    hoverinfo="skip",
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=[0.0, span_x, span_x, 0.0, 0.0],
                    y=[0.0, 0.0, span_y, span_y, 0.0],
                    z=[0, 0, 0, 0, 0],
                    mode="lines",
                    line=dict(color="rgba(139, 0, 0, 1.0)", width=8),
                    name="Court Boundary (AABB fallback)",
                    showlegend=showlegend,
                    hoverinfo="skip",
                )
            )
        traces.append(
            go.Scatter3d(
                x=[0.0, span_x],
                y=[net_d, net_d],
                z=[0, 0],
                mode="lines",
                line=dict(color="rgba(220, 20, 60, 1.0)", width=6),
                name="Net (top view)",
                showlegend=showlegend,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[0.0, span_x],
                y=[nvz_top_d, nvz_top_d],
                z=[0, 0],
                mode="lines",
                line=dict(color="rgba(255, 165, 0, 1.0)", width=6),
                name="Kitchen Line",
                showlegend=showlegend,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[0.0, span_x],
                y=[nvz_bot_d, nvz_bot_d],
                z=[0, 0],
                mode="lines",
                line=dict(color="rgba(255, 165, 0, 1.0)", width=6),
                name="" if showlegend else None,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[cx_d, cx_d],
                y=[0.0, nvz_top_d],
                z=[0, 0],
                mode="lines",
                line=dict(color="rgba(80, 80, 80, 1.0)", width=4),
                name="Service Centerline",
                showlegend=showlegend,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[cx_d, cx_d],
                y=[nvz_bot_d, span_y],
                z=[0, 0],
                mode="lines",
                line=dict(color="rgba(80, 80, 80, 1.0)", width=4),
                name="" if showlegend else None,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        return traces

    if chunk_size <= 0:
        chunk_size = len(ordered_frames)

    frame_chunks = [
        ordered_frames[i : i + chunk_size]
        for i in range(0, len(ordered_frames), chunk_size)
    ]

    base_output_dir = os.path.dirname(os.path.abspath(output_file))
    base_name, ext = os.path.splitext(os.path.basename(output_file))
    if not ext:
        ext = ".html"
    os.makedirs(base_output_dir, exist_ok=True)

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
    }

    for chunk_idx, chunk_frames in enumerate(frame_chunks, start=1):
        chunk_start = chunk_frames[0]
        chunk_end = chunk_frames[-1]

        # Initial frame
        first_frame = chunk_frames[0]
        init_traces = _court_traces(showlegend=True)
        first_idx = 0
        trail_start_idx = max(0, first_idx - 9)
        trail_points = []
        for tfid in chunk_frames[trail_start_idx : first_idx + 1]:
            if tfid in selected_by_frame:
                trail_points.append(selected_by_frame[tfid][1])

        if trail_points:
            tx = [p[0] for p in trail_points]
            ty = [p[1] for p in trail_points]
            tz = [p[2] for p in trail_points]
            init_traces.append(
                go.Scatter3d(
                    x=[x - ox for x in tx],
                    y=[y - oy for y in ty],
                    z=tz,
                    mode="markers",
                    marker=dict(size=5, color="rgb(0, 120, 255)", symbol="circle"),
                    name="Selected Point (last 10 frames)",
                    showlegend=True,
                    customdata=np.column_stack((tx, ty, tz)),
                    hovertemplate=MARKER_HOVER_WORLD,
                )
            )
        else:
            init_traces.append(
                go.Scatter3d(
                    x=[],
                    y=[],
                    z=[],
                    mode="markers",
                    marker=dict(size=5, color="rgb(0, 120, 255)", symbol="circle"),
                    name="Selected Point (last 10 frames)",
                    showlegend=True,
                    visible=False,
                    hoverinfo="skip",
                )
            )

        fig = go.Figure(data=init_traces)

        frames = []
        for idx, fid in enumerate(chunk_frames):
            frame_traces = _court_traces(showlegend=False)
            trail_start_idx = max(0, idx - 9)
            trail_points = []
            for tfid in chunk_frames[trail_start_idx : idx + 1]:
                if tfid in selected_by_frame:
                    trail_points.append(selected_by_frame[tfid][1])

            if trail_points:
                tx = [p[0] for p in trail_points]
                ty = [p[1] for p in trail_points]
                tz = [p[2] for p in trail_points]
                frame_traces.append(
                    go.Scatter3d(
                        x=[x - ox for x in tx],
                        y=[y - oy for y in ty],
                        z=tz,
                        mode="markers",
                        marker=dict(size=5, color="rgb(0, 120, 255)", symbol="circle"),
                        name="Selected Point (last 10 frames)",
                        showlegend=False,
                        visible=True,
                        customdata=np.column_stack((tx, ty, tz)),
                        hovertemplate=MARKER_HOVER_WORLD,
                    )
                )
            else:
                frame_traces.append(
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode="markers",
                        marker=dict(size=5, color="rgb(0, 120, 255)", symbol="circle"),
                        name="Selected Point (last 10 frames)",
                        showlegend=False,
                        visible=False,
                        hoverinfo="skip",
                    )
                )

            frames.append(
                go.Frame(
                    data=frame_traces,
                    layout=dict(
                        title_text=f"{title} - Frame {fid} / {chunk_end}",
                    ),
                    name=str(fid),
                )
            )

        fig.frames = frames

        slider_steps = [
            {
                "method": "animate",
                "args": [[str(fid)], {"frame": {"duration": 33, "redraw": True}, "mode": "immediate", "transition": {"duration": 16}}],
                "label": str(fid),
            }
            for fid in chunk_frames
        ]

        scene_cx = span_x / 2.0
        scene_cy = span_y / 2.0
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(
                    title="X (m) display, +right",
                    range=axis_limits["x"],
                    showgrid=True,
                    gridcolor="lightgray",
                    autorange=False,
                ),
                yaxis=dict(
                    title="Y (m) display, +down",
                    range=axis_limits["y"],
                    showgrid=True,
                    gridcolor="lightgray",
                    autorange=False,
                ),
                zaxis=dict(
                    title="Z (m) display, +up",
                    range=axis_limits["z"],
                    showgrid=True,
                    gridcolor="lightgray",
                    autorange=False,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.0),
                    center=dict(x=scene_cx, y=scene_cy, z=0),
                    up=dict(x=0, y=0, z=1),
                ),
                bgcolor="rgba(240, 240, 240, 0.1)",
            ),
            title=dict(
                text=f"{title}<br><sub>Frames {chunk_start} to {chunk_end} | display origin top-left (min world X,Y); hover shows world XYZ</sub>",
                x=0.5,
                font=dict(size=16),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        {"label": "▶ Play (30 FPS)", "method": "animate", "args": [None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True, "transition": {"duration": 16}}]},
                        {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                    ],
                    pad={"r": 10, "t": 70},
                    x=0.02,
                    y=1.1,
                    xanchor="left",
                    yanchor="top",
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "", "visible": False},
                    pad={"t": 50},
                    steps=slider_steps,
                )
            ],
            width=1200,
            height=800,
            uirevision="constant",
        )

        html_div = pyo.plot(fig, output_type="div", config=config)
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
{html_div}
<script>
window.addEventListener('load', function() {{
  setTimeout(function() {{
    var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
    if (plotDiv) {{
      Plotly.animate(plotDiv, null, {{
        frame: {{duration: 33, redraw: true}},
        transition: {{duration: 16}},
        mode: 'afterall'
      }});
    }}
  }}, 500);
}});
</script>
</body>
</html>
"""
        chunk_output = os.path.join(
            base_output_dir, f"{base_name}_chunk_{chunk_idx:04d}{ext}"
        )
        with open(chunk_output, "w", encoding="utf-8") as f:
            f.write(html)

        print(
            f"[create_html_from_jsonl] Wrote HTML chunk {chunk_idx}/{len(frame_chunks)}: "
            f"{chunk_output} (frames {chunk_start}..{chunk_end}, count={len(chunk_frames)})"
        )

    print(
        f"[create_html_from_jsonl] Total frames={len(ordered_frames)} "
        f"(dense range {lo}..{hi}; sparse JSONL lines had {len(ordered_sparse)} unique ids), "
        f"selected_points={len(selected_by_frame)}, chunks={len(frame_chunks)}, "
        f"chunk_size={chunk_size}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create 3D HTML from trajectory_selection.jsonl")
    p.add_argument(
        "--jsonl-path",
        default=DEFAULT_JSONL_PATH,
        help=f"Path to trajectory_selection.jsonl (default: {DEFAULT_JSONL_PATH})",
    )
    p.add_argument(
        "--output-file",
        default=DEFAULT_HTML_OUTPUT,
        help=(
            f"Output HTML base path (default: {DEFAULT_HTML_OUTPUT}). "
            "Chunk suffixes are appended as _chunk_0001.html, _chunk_0002.html, ..."
        ),
    )
    p.add_argument(
        "--world-points-file",
        default=DEFAULT_WORLD_POINTS_PATH,
        help=(
            "Path to world points txt (default tries tools/pickleball_calib/worldpickleball.txt, "
            "then calibration_testing/source/world_points.txt). "
            "Each line: x y z or name x y z (comments with #)."
        ),
    )
    p.add_argument("--title", default="3D Selected Point Visualization")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max frames per output HTML chunk (default: 1000). Use <=0 for single file.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    create_3d_trajectory_visualization_from_jsonl(
        jsonl_path=args.jsonl_path,
        output_file=args.output_file,
        world_points_file=args.world_points_file,
        title=args.title,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
