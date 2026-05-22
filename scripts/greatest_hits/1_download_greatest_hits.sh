#!/usr/bin/env bash
# ============================================================
# Greatest Hits Dataset Downloader
# Source: https://andrewowens.com/vis/
# ============================================================

set -euo pipefail

BASE_URL_UMICH="https://web.eecs.umich.edu/~ahowens/vis"
BASE_URL_OWENS="https://andrewowens.com/vis"

FULL_RES="vis-data.zip"          # ~50 GB  — full-resolution videos + labels
LOW_RES="vis-data-256.zip"       # ~20 GB  — 456×256 videos + labels
FEATURES="vis-sfs.zip"           # ~1  GB  — precomputed sound features

# ---------- helpers ----------------------------------------------------------
usage() {
  cat <<USAGE
Usage: $0 [OPTIONS]

Options:
  --full         Download full-resolution videos + labels  (~50 GB)
  --low          Download low-resolution  videos + labels  (~20 GB)
  --features     Download precomputed sound features       (~ 1 GB)
  --all          Download everything
  --dir DIR      Target directory (default: <script>/../data/greatest_hits)
  --scratch      Download to Snellius scratch-local: /scratch-local/<user>/greatest_hits
  -h, --help     Show this help

Examples:
  $0 --low                         # recommended starting point
  $0 --all --dir /data/vis
  $0 --low --scratch               # download to /scratch-local/<user>/greatest_hits
USAGE
  exit 0
}

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

download() {
  local url="$1" dest="$2"
  local filename
  filename="$(basename "$url")"
  local filepath="${dest}/${filename}"

  # Validate existing file, re-download if corrupted
  if [[ -f "$filepath" ]]; then
    log "Validating existing file: $filename ..."
    if unzip -t "$filepath" &>/dev/null; then
      log "File is valid, skipping download: $filename"
    else
      log "File is corrupted, deleting and re-downloading: $filename"
      rm -f "$filepath"
    fi
  fi

  if [[ ! -f "$filepath" ]]; then
    log "Downloading $filename ..."
    if command -v wget &>/dev/null; then
      wget --no-check-certificate --progress=bar:force:noscroll -c -O "$filepath" "$url"
    elif command -v curl &>/dev/null; then
      curl -L -k --progress-bar -C - -o "$filepath" "$url"
    else
      die "Neither wget nor curl found. Please install one of them."
    fi
    log "Saved to: $filepath"
  fi

  # Unzip
  local extract_dir="${dest}/${filename%.zip}"
  if [[ -d "$extract_dir" ]]; then
    log "Already extracted, skipping unzip: $filename"
  else
    log "Extracting $filename ..."
    if command -v unzip &>/dev/null; then
      unzip -q "$filepath" -d "$extract_dir"
    else
      die "unzip not found. Please install it (e.g. apt install unzip)."
    fi
    log "Extracted to: $extract_dir"
    rm -f "$filepath"
    log "Removed zip: $filename"
  fi
}

# ---------- argument parsing -------------------------------------------------
DO_FULL=false
DO_LOW=false
DO_FEATURES=false
USE_SCRATCH=false
SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
DEST_DIR="$(dirname "$SCRIPT_PATH")/../../data/greatest_hits"

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)     DO_FULL=true ;;
    --low)      DO_LOW=true ;;
    --features) DO_FEATURES=true ;;
    --all)      DO_FULL=true; DO_LOW=true; DO_FEATURES=true ;;
    --dir)      shift; DEST_DIR="$1" ;;
    --scratch)  USE_SCRATCH=true ;;
    -h|--help)  usage ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

# --scratch overrides --dir; resolve username at runtime via $(whoami)
if [[ "$USE_SCRATCH" == true ]]; then
  SCRATCH_USER="$(whoami)"
  SCRATCH_BASE="/scratch-local/${SCRATCH_USER}"
  if [[ ! -d "$SCRATCH_BASE" ]]; then
    die "Scratch-local directory not found: $SCRATCH_BASE — are you on a Snellius node?"
  fi
  DEST_DIR="${SCRATCH_BASE}/greatest_hits"
  log "Scratch mode enabled → using /scratch-local/${SCRATCH_USER}/greatest_hits"
  log "WARNING: scratch-local files older than 6 days are deleted automatically. Copy results to your home or project space when done."
fi

# ---------- download ---------------------------------------------------------
mkdir -p "$DEST_DIR"
log "Destination: $(realpath "$DEST_DIR")"

$DO_FEATURES && download "${BASE_URL_OWENS}/${FEATURES}"  "$DEST_DIR"
$DO_LOW      && download "${BASE_URL_UMICH}/${LOW_RES}"   "$DEST_DIR"
$DO_FULL     && download "${BASE_URL_UMICH}/${FULL_RES}"  "$DEST_DIR"

log "Done."