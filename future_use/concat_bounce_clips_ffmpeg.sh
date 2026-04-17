#!/usr/bin/env bash
# Concatenate bounce clips per camera: one MP4 for all source clips, one for all sink.
# Does NOT merge source+sink into a single file — two separate concatenations only.
#
# Usage:
#   ./concat_bounce_clips_ffmpeg.sh [BOUNCE_CLIPS_ROOT]
#
# Default BOUNCE_CLIPS_ROOT: /mnt/data/cv_output/bounce_clips
# Expects:  <root>/source/*.mp4  and  <root>/sink/*.mp4
# Writes:   <root>/source_concat.mp4  and  <root>/sink_concat.mp4
#
# Requires: ffmpeg on PATH. Clips should share codec/size for concat demuxer; if concat
# fails, re-encode with: ffmpeg -f concat -safe 0 -i list.txt -c:v libx264 -crf 18 out.mp4

set -euo pipefail

ROOT="${1:-/mnt/data/cv_output/bounce_clips}"

if [[ ! -d "$ROOT" ]]; then
  echo "Directory not found: $ROOT" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH" >&2
  exit 1
fi

concat_one() {
  local subdir="$1"
  local out_name="$2"
  local dir="$ROOT/$subdir"
  local out="$ROOT/$out_name"

  if [[ ! -d "$dir" ]]; then
    echo "[concat] skip: no directory $dir"
    return 0
  fi

  mapfile -t files < <(find "$dir" -maxdepth 1 -type f -name '*.mp4' | LC_ALL=C sort -V)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "[concat] skip: no .mp4 under $dir"
    return 0
  fi

  local list
  list="$(mktemp)"
  # ffmpeg concat: one line per file; paths with single quotes need escaping (rare).
  for f in "${files[@]}"; do
    printf "file '%s'\n" "${f//\'/\'\\\'\'}" >>"$list"
  done

  echo "[concat] $subdir -> $out (${#files[@]} files)"
  if ffmpeg -y -hide_banner -loglevel warning -f concat -safe 0 -i "$list" -c copy "$out"; then
    echo "[concat] wrote $out"
  else
    echo "[concat] -c copy failed (codec mismatch?). Try re-encoding, e.g.:" >&2
    echo "  ffmpeg -f concat -safe 0 -i \"$list\" -c:v libx264 -crf 18 -c:a aac \"$out\"" >&2
    rm -f "$list"
    return 1
  fi
  rm -f "$list"
}

concat_one "source" "source_concat.mp4"
concat_one "sink" "sink_concat.mp4"
echo "[concat] done."
