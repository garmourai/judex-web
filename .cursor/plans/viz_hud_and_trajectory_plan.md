# Trajectory visualization and selection — consolidated plan

## Scope (five workstreams)

1. **Right cost panel layout** — fix overlapping text / collision with left HUD in [`triangulation.py`](cv_code/correlation/visualization/triangulation.py).
2. **Decision overlap across segments** — extend `get_best_point_each_frame` so pass `[x, y]` decides **`x−10 … y−10`** (recover the previous segment’s undecided tail), in [`select_best.py`](cv_code/correlation/trajectory/select_best.py).
3. **Dense video output** — emit **every** frame in `[f_min, f_max]` with camera image + HUD; overlay only when data exists; **no gaps** in the MP4.
4. **Legacy classic `StagingBuffer` removal** — OFB-only judex path; remove or gate dual-path code (Phase 4).
5. **`OriginalFrameBuffer` cleanup retention** — align `clear_batch` threshold with **`y−10`** on non-final segments so the last 10 frames stay for overlap drawing; on **final** segment clear through **`y`** (Phase 5).

---

## Phase 1: Right “Propagated Costs” panel (layout)

**Problem:** Fixed `box_h` and tiny `cost_line_spacing` vs real `getTextSize` heights causes title and F1…Temp lines to draw on top of each other. On narrow frames, left HUD and right panel can overlap horizontally.

**Implementation:**

- Measure each row with `cv2.getTextSize` (title scale vs line scale), stack baselines with `row_height = text_h + gap`.
- Compute `box_w` / `box_h` from measured extents + padding; draw the semi-transparent rectangle **after** measuring.
- If `x1 < 10 + hud_w + gap`, move the cost block **below** the left HUD (`y1 = 10 + overlay_height + gap`), keep right-aligned (`x2 = W - right_pad`).

**Files:** [`cv_code/correlation/visualization/triangulation.py`](cv_code/correlation/visualization/triangulation.py)

---

## Phase 2: Segment tail overlap in `get_best_point_each_frame`

**Model:** For previous segment `[a, x−1]`, decisions stop at **`(x−1) − 10 = x − 11`**. Frames **`x−10 … x−1`** are intentionally undecided there. Current segment **`[x, y]`** must **decide those 10 frames** in addition to **`x … y−10`**.

**Implementation:**

- Let `start_frame, end_frame = segment`, `L = LAST_FRAMES_TO_SKIP` (10).
- **`decision_low = max(0, start_frame - L)`** (or `max(first_frame_with_data, …)` if you add a guard).
- **`decision_high = end_frame - L`** (unchanged upper bound).
- **`frames_to_process = sorted(f for f in frame_to_trajectories.keys() if decision_low <= f <= decision_high)`**.

**Prerequisites:** Trajectories passed into `get_best_point_each_frame` must still contain detections for **`x−10 … x−1`** (typically **handoff** from [`tracker_tree` / `handoff_context`](cv_code/correlation/trajectory/realtime/handoff_context.py)). If handoff does not carry those frames, overlap logic will not emit rows for them — verify on a real segment boundary.

**JSONL:** Rows for `f in [x−10, x−1]` will be appended in the **same** batch as segment `[x, y]` (same `segment` field in JSON as today, or optionally add `decision_window` metadata later).

**Files:** [`cv_code/correlation/trajectory/select_best.py`](cv_code/correlation/trajectory/select_best.py)

```mermaid
flowchart LR
  prev[Pass segment a to x-1]
  curr[Pass segment x to y]
  prev -->|decides a..x-11| gap[Undecided x-10..x-1]
  gap -->|curr pass decides| curr
  curr -->|decides x-10..y-10| next[Next pass covers y-9..]
```

---

## Phase 3: Dense frame range in visualization

**Goal:** For **`[f_min, f_max]`** (see below), output **exactly** `f_max - f_min + 1` frames — **no skipping** when JSONL has no row or no 3D points.

**Choose `[f_min, f_max]`** (recommend aligning with Phase 2):

- **`f_min = segment_start - LAST_FRAMES_TO_SKIP`** (same `x−10` idea), capped at 0.
- **`f_max = segment_end`** (include full segment in the video timeline), **or** `segment_end` only — product choice: including tail frames with HUD-only is usually desired so the MP4 matches correlation batch length.

**Implementation in [`triangulation.py`](cv_code/correlation/visualization/triangulation.py):**

1. After building maps from JSONL, compute `f_min`, `f_max` from `frame_segments` + constant `L` (import or duplicate `LAST_FRAMES_TO_SKIP` from select_best or a shared small constant module to avoid drift).
2. **Load JSONL** for all frames that might appear: either widen `_load_trajectory_selection_jsonl` filter to **`[f_min, f_max]`** (not only strict segment), or load without lower filter when `frame_ranges` extended — ensure **`x−10 …`** lines from Phase 2 are loaded when viz window includes them.
3. Replace `for frame_idx in sorted(all_frame_data.keys())` with **`for frame_idx in range(f_min, f_max + 1)`**.
4. For each `frame_idx`:
   - Resolve `coords_list` / trajectory maps via `.get()` — default **empty** / no overlay.
   - Fetch `frame_img` from buffer (same mapping as today). If **`None`**: use **solid black** image of same size as first good frame, or first available frame size from config (`original_frame_width/height` if passed in, else skip-with-black policy documented).
   - Always draw **HUD** (counts can be 0).
   - Append one entry to `frames_with_points`.
5. **MP4 filename:** `trajectory_visualization_{f_min}_to_{f_max}.mp4` for each chunk, or chunk subranges `chunk[0][0]_to_chunk[-1][0]` if `video_chunk_size` splits — document that chunk names are **contiguous subranges** of `[f_min, f_max]`.

**Worker:** [`correlation_worker.py`](cv_code/correlation/correlation_worker.py) — pass **`frame_segments`** (or a new optional `visualization_frame_range`) that matches **`(f_min, f_max)`** for `create_visualization_from_triangulation` so JSONL filtering includes overlap rows.

**Files:** [`cv_code/correlation/visualization/triangulation.py`](cv_code/correlation/visualization/triangulation.py), [`cv_code/correlation/correlation_worker.py`](cv_code/correlation/correlation_worker.py)

---

## Ordering / dependencies

| Order | Phase | Depends on |
|------|--------|------------|
| 1 | Cost panel layout | None |
| 2 | `frames_to_process` overlap | None (verify handoff has data) |
| 3 | Dense viz | Phase 2 helps JSONL exist for `x−10…`; Phase 1 keeps HUD readable on every frame |
| 4 | OFB-only / legacy removal | Independent; run after Phases 1–3 if viz/worker signatures change |
| 5 | OFB `clear_batch` retention | After Phase 3 dense viz; pairs with Phase 2 overlap (pixels for `y−9…y` until next pass) |

---

## Phase 5: `OriginalFrameBuffer` cleanup — retain tail until next segment

**Goal:** The last **`L = 10`** frames of segment **`[x, y]`** (`y−9 … y`) are needed for **dense drawing** and **overlap** with the next segment’s decisions (`x_next−10 …`). Today [`correlation_worker.py`](cv_code/correlation/correlation_worker.py) clears reader batches when **`peek_batch_source_index_range`’s `max_source_index <= segment[1]`** (i.e. **`y`**). Any batch fully ending at **`≤ y`** is removed **whole** — which can **drop pixels for `y−9 … y`** before the next correlation pass runs, hurting consistent drawing.

**Policy:**

- **Non-final** correlation segment: use cleanup cutoff **`retention_end = y − L`** (i.e. **`y − 10`**). Clear only batches with **`_max_src <= retention_end`** (same loop structure as today, different threshold). Frames **`y−9 … y`** (and any batch that still spans past **`y−10`**) remain in `OriginalFrameBuffer` for the next segment’s viz / overlap.
- **Final** segment (end of run): use cutoff **`y`** so buffers drain — **`_max_src <= y`** (current behavior) or **`clear_all`** if simpler after last viz.

**“Final segment” signal:** e.g. after processing, **`inference_done_event.is_set()`** (or equivalent) **and** no further frames expected — must be wired from [`triplet_pipeline_runner.py`](cv_code/triplet_pipeline_runner.py) / worker contract (new optional `bool` or reuse existing events). Document edge case: worker exits mid-run → treat like final for cleanup.

**Granularity caveat:** `clear_batch` removes **entire reader batches**, not single frames. Retention is **batch-level**: a batch whose **`max_src > y−10`** is **not** cleared mid-segment; it may be cleared once a **later** segment’s cutoff passes its `max_src`. This matches “keep anything that touches the tail” without a per-frame delete API (unless you add one later).

**Constant:** Reuse **`LAST_FRAMES_TO_SKIP`** (10) from [`select_best.py`](cv_code/correlation/trajectory/select_best.py) or a tiny shared constant to avoid drift.

**Files:** [`cv_code/correlation/correlation_worker.py`](cv_code/correlation/correlation_worker.py), possibly [`cv_code/triplet_pipeline_runner.py`](cv_code/triplet_pipeline_runner.py) for final-segment flag.

---

## Verification

- **Phase 1:** One narrow (640px) and one wide frame — cost block readable, no overlap with left HUD.
- **Phase 2:** At segment boundary, JSONL contains `frame_id` in **`x−10 … x−1`** for the **second** segment batch; no stray pre-`a` frames unless handoff includes them.
- **Phase 3:** Frame count in MP4 equals **`f_max - f_min + 1`** (per chunk sum if chunked); spot-check a frame with no JSONL still present with HUD.
- **Phase 4:** Triplet/judex run unchanged after OFB-only cleanup; grep shows no required classic staging path for overlay.
- **Phase 5:** After non-final segment, OFB still returns frames for **`y−9…y`** (or batch containing them); after final segment, buffer cleared through **`y`**.

---

## Todos

- [ ] **hud-cost-layout** — Measured vertical layout + collision avoid (below left HUD) in triangulation.py
- [ ] **select-best-overlap** — `decision_low = max(0, start_frame - LAST_FRAMES_TO_SKIP)` through `end_frame - LAST_FRAMES_TO_SKIP`
- [ ] **viz-dense-range** — `range(f_min, f_max+1)`, empty overlay + black on miss, JSONL load window, worker passes extended range
- [ ] **verify-boundary** — Manual check segment boundary JSONL + frame count
- [ ] **legacy-audit** — Confirm no non-triplet entry points use classic `StagingBuffer`
- [ ] **ofb-only-cleanup** — Remove or gate legacy branches; shrink `staging_cleanup.py` + exports
- [ ] **ofb-clear-retention** — Non-final: clear batches with `max_src <= end_frame - 10`; final: clear through `end_frame`; wire `is_final_segment`

---

## Phase 4: Legacy classic `StagingBuffer` paths — audit / removal

**Current production path (judex):** [`triplet_pipeline_runner.py`](cv_code/triplet_pipeline_runner.py) passes [`OriginalFrameBuffer`](cv_code/triplet_csv_reader.py) as `staging_buffer_1` into `correlation_worker`. `correlation_worker` is not started from other modules in-repo (only the wrapper).

**Legacy branches still present (dead for judex, live if someone passes classic buffers):**

| Location | What it does |
|----------|----------------|
| [`triangulation.py`](cv_code/correlation/visualization/triangulation.py) | `else` branch: `load_fid_to_stream_from_dist_tracker_csv`, `peek_frames_by_indices`, requires `camera_1_csv_path` |
| [`correlation_worker.py`](cv_code/correlation/correlation_worker.py) | `else` after `OriginalFrameBuffer` check: `cleanup_staging_buffers_from_triangulation` |
| [`stitched.py`](cv_code/correlation/visualization/stitched.py) | `else`: dual `peek_frames_by_indices` on two buffers when not “same OFB for both cams” |
| [`staging_cleanup.py`](cv_code/correlation/trajectory/staging_cleanup.py) | Timestamp-threshold removal for classic `StagingBuffer` |
| [`trajectory/__init__.py`](cv_code/correlation/trajectory/__init__.py) | Re-exports `cleanup_staging_buffers_from_triangulation` |

**Cleanup approach (if you commit to OFB-only):**

1. Require `OriginalFrameBuffer` (or a small protocol) in viz + worker; drop `else` branches and `staging_cleanup` usage from worker.
2. Simplify `append_stitched_segment_to_video` to OFB-only or keep a thin adapter.
3. Remove or archive `staging_cleanup.py` and exports if nothing else imports it.
4. Grep for `peek_frames_by_indices`, `remove_frames_by_timestamp_threshold` before deleting.

**Risk:** External scripts or forks that still inject classic `StagingBuffer` would break unless kept behind a feature flag.

---
