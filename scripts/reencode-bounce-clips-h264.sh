#!/usr/bin/env bash
# Re-encode bounce clip MP4s under events/bounce_clips/ to H.264 + AAC + faststart
# so they play in Chrome (<video> does not support MPEG-4 Part 2 / mp4v).
#
# Usage (from repo root):
#   chmod +x scripts/reencode-bounce-clips-h264.sh
#   ./scripts/reencode-bounce-clips-h264.sh
#   ./scripts/reencode-bounce-clips-h264.sh --dry-run
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLIP_ROOT="${ROOT}/events/bounce_clips"
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install: sudo apt install ffmpeg" >&2
  exit 1
fi

if [[ ! -d "$CLIP_ROOT" ]]; then
  echo "No directory: $CLIP_ROOT" >&2
  exit 1
fi

count=0
while IFS= read -r -d '' f; do
  count=$((count + 1))
  vcodec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" 2>/dev/null || echo "")"
  if [[ "$vcodec" == "h264" ]]; then
    echo "skip (already h264): $f"
    continue
  fi

  tmp="${f}.reencode.tmp.mp4"
  echo "encode: $f (video codec: ${vcodec:-unknown})"

  if [[ "$DRY" -eq 1 ]]; then
    continue
  fi

  if ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$f" 2>/dev/null | grep -q .; then
    ffmpeg -hide_banner -loglevel warning -y -i "$f" \
      -c:v libx264 -crf 23 -preset fast -movflags +faststart \
      -c:a aac -b:a 128k \
      "$tmp"
  else
    ffmpeg -hide_banner -loglevel warning -y -i "$f" \
      -c:v libx264 -crf 23 -preset fast -movflags +faststart \
      -an \
      "$tmp"
  fi

  mv "$tmp" "$f"
  echo "  -> done"
done < <(find "$CLIP_ROOT" -type f -name '*.mp4' -print0)

if [[ "$count" -eq 0 ]]; then
  echo "No .mp4 files under $CLIP_ROOT"
else
  echo "Finished ($count file(s) processed or skipped)."
fi
