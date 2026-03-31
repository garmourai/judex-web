#!/usr/bin/env bash
#
# Rsync ts_segments/<track_id>/ from HQ (SOURCE_IPS_CHECK) and sink (SINK_IPS) Pis
# into /home/pi/source_code/ts_segments_hq/<track_id>/ and ts_segments_sink/<track_id>/ on this machine.
#
#   track_id_index  →  variable_files/track_video_index.json  ("counter")
#   HQ / sink IPs   →  variable_files/config.yaml
#
set -euo pipefail
shopt -s extglob

trim_ws() {
  local s="${1-}"
  s="${s##+([[:space:]])}"
  s="${s%%+([[:space:]])}"
  printf '%s' "${s}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIABLES_DIR="${VARIABLES_DIR:-/home/pi/source_code/variable_files}"
CONFIG_YAML="${CONFIG_YAML:-${VARIABLES_DIR}/config.yaml}"
TRACK_INDEX_JSON="${TRACK_INDEX_JSON:-${VARIABLES_DIR}/track_video_index.json}"
DEST_HQ="${DEST_HQ:-/home/pi/source_code/ts_segments_hq}"
DEST_SINK="${DEST_SINK:-/home/pi/source_code/ts_segments_sink}"
SSH_USER="${SSH_USER:-pi}"
SOURCE_BASE="${SOURCE_BASE:-/home/pi/source_code/ts_segments}"
SINK_BASE="${SINK_BASE:-/home/pi/sink_code/ts_segments}"
DRY_RUN=0
TRACK_ID_OVERRIDE=""
RUN_HQ=0
RUN_SINK=0

usage() {
  echo "Usage: $(basename "$0") [--variables-dir <dir>] [--config <config.yaml>]"
  echo "                         [--track-json <track_video_index.json>]"
  echo "                         [--dest-hq <dir>] [--dest-sink <dir>] [--user <ssh_user>]"
  echo "                         [--source-base <remote_dir>] [--sink-base <remote_dir>]"
  echo "                         [--track-id <id>] [--dry-run]"
  echo "                         [--hq] [--sink]"
  echo
  echo "Defaults:"
  echo "  --variables-dir  ${VARIABLES_DIR}"
  echo "  --dest-hq        ${DEST_HQ}"
  echo "  --dest-sink      ${DEST_SINK}"
  echo "  --user           ${SSH_USER}"
  echo "  --source-base    ${SOURCE_BASE}"
  echo "  --sink-base      ${SINK_BASE}"
  echo
  echo "  --hq             Only rsync from HQ (SOURCE_IPS_CHECK) servers"
  echo "  --sink           Only rsync from sink (SINK_IPS) servers"
  echo "                   (omit both to rsync all in parallel)"
  echo
  echo "Reads track_id_index (counter) from track_video_index.json and HQ/sink IPs from config.yaml."
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage ;;
    --variables-dir)
      VARIABLES_DIR="${2:-}"; shift 2
      CONFIG_YAML="${VARIABLES_DIR}/config.yaml"
      TRACK_INDEX_JSON="${VARIABLES_DIR}/track_video_index.json"
      ;;
    --config)
      CONFIG_YAML="${2:-}"; shift 2 ;;
    --track-json)
      TRACK_INDEX_JSON="${2:-}"; shift 2 ;;
    --dest-hq)
      DEST_HQ="${2:-}"; shift 2 ;;
    --dest-sink)
      DEST_SINK="${2:-}"; shift 2 ;;
    --user)
      SSH_USER="${2:-}"; shift 2 ;;
    --source-base)
      SOURCE_BASE="${2:-}"; shift 2 ;;
    --sink-base)
      SINK_BASE="${2:-}"; shift 2 ;;
    --track-id)
      TRACK_ID_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --hq)
      RUN_HQ=1; shift ;;
    --sink)
      RUN_SINK=1; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

load_config_and_track() {
  export CONFIG_YAML TRACK_INDEX_JSON TRACK_ID_OVERRIDE
  python3 -c "
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print('Python PyYAML is required (pip install pyyaml).', file=sys.stderr)
    sys.exit(1)

cfg_path = Path(os.environ['CONFIG_YAML'])
idx_path = Path(os.environ['TRACK_INDEX_JSON'])

if not cfg_path.is_file():
    print(f'Config not found: {cfg_path}', file=sys.stderr)
    sys.exit(1)
if not idx_path.is_file():
    print(f'Track index not found: {idx_path}', file=sys.stderr)
    sys.exit(1)

with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

with open(idx_path, 'r', encoding='utf-8') as f:
    tj = json.load(f)

ov = os.environ.get('TRACK_ID_OVERRIDE', '').strip()
track_id = int(ov) if ov else int(tj.get('counter', 0))

def ips(key):
    v = cfg.get(key) or []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None:
        return []
    return [str(v).strip()]

hq = ips('SOURCE_IPS_CHECK')
sinks = ips('SINK_IPS')

print(track_id)
print(','.join(hq))
print(','.join(sinks))
"
}

mapfile -t _lines < <(load_config_and_track)
TRACK_ID="${_lines[0]}"
IFS=',' read -r -a HQ_IPS <<< "${_lines[1]:-}"
IFS=',' read -r -a SINK_IPS <<< "${_lines[2]:-}"

remote_dir_for_ip() {
  local ip="$1"
  if [[ "${ip}" == "${SINK_IP_FOR_REMOTE:-}" ]]; then
    echo "${SINK_BASE%/}/${TRACK_ID}"
  else
    echo "${SOURCE_BASE%/}/${TRACK_ID}"
  fi
}

remote_dropped_csv_for_ip() {
  local ip="$1"
  local base="/home/pi/source_code/streamed_packets"
  if [[ "${ip}" == "${SINK_IP_FOR_REMOTE:-}" ]]; then
    base="/home/pi/sink_code/streamed_packets"
  fi
  echo "${base%/}/${TRACK_ID}/local_ts_dropped_packets.csv"
}

dest_dir_for() {
  local is_sink="$1"
  if [[ "${is_sink}" -eq 1 ]]; then
    echo "${DEST_SINK%/}/${TRACK_ID}"
  else
    echo "${DEST_HQ%/}/${TRACK_ID}"
  fi
}

ssh_dir_ok() {
  local ip="$1"
  local rdir="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    "${SSH_USER}@${ip}" "test -d \"${rdir}\"" >/dev/null 2>&1
}

print_temp_for_ip() {
  local ip="$1"
  local temp_script_primary="/home/pi/source_code/check_temp.sh"
  local temp_script_fallback="/home/pi/sink_code/check_temp.sh"

  set +e
  out="$(
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
      "${SSH_USER}@${ip}" "if [[ -f \"${temp_script_primary}\" ]]; then bash \"${temp_script_primary}\"; elif [[ -f \"${temp_script_fallback}\" ]]; then bash \"${temp_script_fallback}\"; else echo \"missing: ${temp_script_primary}\" 1>&2; exit 127; fi" 2>&1
  )"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "Temp: ${ip} (failed: ${out})" >&2
    return 1
  fi

  out="${out//$'\r'/}"
  out="${out//$'\n'/ }"
  out="$(trim_ws "${out}")"
  if [[ -z "${out}" ]]; then
    echo "Temp: ${ip} (no output)" >&2
    return 1
  fi

  echo "Temp: ${ip} ${out}"
}

RSYNC_SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# If neither --hq nor --sink given, run both
if [[ "${RUN_HQ}" -eq 0 && "${RUN_SINK}" -eq 0 ]]; then
  RUN_HQ=1
  RUN_SINK=1
fi

mkdir -p "${DEST_HQ%/}/${TRACK_ID}" "${DEST_SINK%/}/${TRACK_ID}"

# Temp dir for communicating results from background jobs back to parent
RESULT_DIR="$(mktemp -d)"
trap 'rm -rf "${RESULT_DIR}"' EXIT

pull_one() {
  local ip="$1"
  local name="$2"
  local is_sink="$3"
  local result_file="$4"

  ip="$(trim_ws "${ip}")"
  [[ -z "${ip}" ]] && { echo "skip" > "${result_file}"; return 0; }

  if [[ "${is_sink}" -eq 1 ]]; then
    SINK_IP_FOR_REMOTE="${ip}"
  else
    SINK_IP_FOR_REMOTE=""
  fi

  local LABEL="${name} (${ip})"
  local REMOTE_DIR
  REMOTE_DIR="$(remote_dir_for_ip "${ip}")"
  local LOCAL_DIR
  LOCAL_DIR="$(dest_dir_for "${is_sink}")"
  mkdir -p "${LOCAL_DIR}"

  if ! ssh_dir_ok "${ip}" "${REMOTE_DIR}"; then
    echo "Skip:  ${ip} (unreachable or missing: ${REMOTE_DIR})" >&2
    echo "fail" > "${result_file}"
    return 0
  fi

  print_temp_for_ip "${ip}" || true

  echo "Name:  ${LABEL}"
  echo "Pull:  ${SSH_USER}@${ip}:${REMOTE_DIR}/"
  echo "Into:  ${LOCAL_DIR}/"
  set +e
  rsync_args=(
    -a
    --itemize-changes
    --partial
    --partial-dir=".rsync-partial"
    --timeout=30
    -e "ssh ${RSYNC_SSH_OPTS}"
  )
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    rsync_args+=(--dry-run)
  fi
  rsync_output="$(rsync "${rsync_args[@]}" \
    "${SSH_USER}@${ip}:${REMOTE_DIR}/" \
    "${LOCAL_DIR}/" 2>&1)"
  rc=$?
  # Show first 2 transferred files (lines starting with >f = new/changed file)
  transferred="$(echo "${rsync_output}" | grep '^>f' | head -n 2)"
  total_new="$(echo "${rsync_output}" | grep -c '^>f' || true)"
  if [[ -n "${transferred}" ]]; then
    echo "New:   ${transferred}"
    if [[ "${total_new}" -gt 2 ]]; then
      echo "       ... and $((total_new - 2)) more files"
    fi
  else
    echo "Sync:  ${LABEL} (no new files)"
  fi
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "Fail:  ${ip} (rsync exit ${rc})" >&2
    echo "fail" > "${result_file}"
    return 0
  fi
  echo "ok" > "${result_file}"

  local REMOTE_DROPPED_CSV
  REMOTE_DROPPED_CSV="$(remote_dropped_csv_for_ip "${ip}")"
  set +e
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    "${SSH_USER}@${ip}" "test -f \"${REMOTE_DROPPED_CSV}\"" >/dev/null 2>&1
  has_csv_rc=$?
  set -e
  if [[ "${has_csv_rc}" -eq 0 ]]; then
    set +e
    csv_rsync_args=(
      -a
      --timeout=30
      -e "ssh ${RSYNC_SSH_OPTS}"
    )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      csv_rsync_args+=(--dry-run)
    fi
    rsync "${csv_rsync_args[@]}" \
      "${SSH_USER}@${ip}:${REMOTE_DROPPED_CSV}" \
      "${LOCAL_DIR}/local_ts_dropped_packets.csv"
    csv_rc=$?
    set -e
    if [[ $csv_rc -ne 0 ]]; then
      echo "Warn: ${LABEL} (failed to pull local_ts_dropped_packets.csv; rsync exit ${csv_rc})" >&2
    fi
  else
    echo "Warn: ${LABEL} (missing: ${REMOTE_DROPPED_CSV})" >&2
  fi

  echo
}

iteration=0

while true; do
  iteration=$((iteration + 1))
  echo "=== Iteration ${iteration} ($(date '+%H:%M:%S')) ==="

  rm -f "${RESULT_DIR}"/*

  pids=()
  job_idx=0

  if [[ "${RUN_HQ}" -eq 1 ]]; then
    for ip in "${HQ_IPS[@]}"; do
      ip="$(trim_ws "${ip}")"
      [[ -z "${ip}" ]] && continue
      pull_one "${ip}" "hq" 0 "${RESULT_DIR}/${job_idx}" &
      pids+=($!)
      job_idx=$((job_idx + 1))
    done
  fi

  if [[ "${RUN_SINK}" -eq 1 ]]; then
    for ip in "${SINK_IPS[@]}"; do
      ip="$(trim_ws "${ip}")"
      [[ -z "${ip}" ]] && continue
      pull_one "${ip}" "sink" 1 "${RESULT_DIR}/${job_idx}" &
      pids+=($!)
      job_idx=$((job_idx + 1))
    done
  fi

  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done

  # Tally results from temp files written by subshells
  pulled=0
  failures=0
  for f in "${RESULT_DIR}"/*; do
    [[ -f "${f}" ]] || continue
    status="$(<"${f}")"
    case "${status}" in
      ok) pulled=$((pulled + 1)) ;;
      fail|skip) failures=$((failures + 1)) ;;
    esac
  done

  if [[ ${job_idx} -eq 0 ]]; then
    echo "No IPs selected (check config or --hq / --sink flags)." >&2
    exit 2
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Iteration ${iteration} done (dry-run). Pulled: ${pulled}. Skipped/failed: ${failures}."
  else
    echo "Iteration ${iteration} done. Pulled: ${pulled}. Failures: ${failures}."
  fi

  echo "--- Sleeping 1s (Ctrl+C to stop) ---"
  sleep 1
done
