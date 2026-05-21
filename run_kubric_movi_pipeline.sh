#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-kubric_movi}"
ENV_YAML="${ENV_YAML:-$SCRIPT_DIR/conda_kubric_movi.yaml}"
DATASET="${DATASET:-movi_a/128x128}"
SPLIT="${SPLIT:-validation}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/kubric_tfds}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/data/kubric_movi_a}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
TOUCH_RADIUS="${TOUCH_RADIUS:-3}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-1}"
NO_TOUCH_FPS="${NO_TOUCH_FPS:-2}"
HARD_NEGATIVE_WINDOW="${HARD_NEGATIVE_WINDOW:-4}"
VIDEO_FPS="${VIDEO_FPS:-12}"

echo "Repository:     $REPO_ROOT"
echo "Environment:    $ENV_NAME"
echo "Environment yml:$ENV_YAML"
echo "TFDS data dir:  $DATA_DIR"
echo "Dataset:        $DATASET"
echo "Split:          $SPLIT"
echo "Output root:    $OUTPUT_ROOT"
echo "Touch radius:   $TOUCH_RADIUS"
echo "Download data:  $DOWNLOAD_DATASET"
echo "No-touch fps:   $NO_TOUCH_FPS"
echo "Hard neg window:$HARD_NEGATIVE_WINDOW"
echo "Video fps:      $VIDEO_FPS"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available on PATH." >&2
  exit 1
fi

if [ ! -f "$ENV_YAML" ]; then
  echo "Error: environment YAML not found: $ENV_YAML" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_YAML" --prune
else
  echo "Creating conda environment: $ENV_NAME"
  conda env create -f "$ENV_YAML"
fi

DATASET_DIR="$DATA_DIR/${DATASET}/1.0.0"

if [ "$DOWNLOAD_DATASET" = "1" ]; then
  if { [ "$DATASET" != "movi_a/128x128" ] && [ "$DATASET" != "movi_a/256x256" ]; } \
      || { [ "$SPLIT" != "train" ] && [ "$SPLIT" != "validation" ]; }; then
    echo "Error: automatic download currently supports DATASET=movi_a/128x128 or movi_a/256x256, with SPLIT=train or validation." >&2
    exit 1
  fi
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "Error: gsutil is not available on PATH; cannot download MOVi-A." >&2
    exit 1
  fi

  mkdir -p "$DATASET_DIR"
  if ls "$DATASET_DIR"/movi_a-"$SPLIT".tfrecord-* >/dev/null 2>&1 \
      && [ -f "$DATASET_DIR/dataset_info.json" ] \
      && [ -f "$DATASET_DIR/features.json" ]; then
    echo "MOVi-A $SPLIT files already present; skipping download."
  else
    echo "Downloading $DATASET $SPLIT split into: $DATASET_DIR"
    gsutil -m cp \
      "gs://kubric-public/tfds/$DATASET/1.0.0/dataset_info.json" \
      "gs://kubric-public/tfds/$DATASET/1.0.0/features.json" \
      "gs://kubric-public/tfds/$DATASET/1.0.0/instances-color_label.labels.txt" \
      "gs://kubric-public/tfds/$DATASET/1.0.0/instances-material_label.labels.txt" \
      "gs://kubric-public/tfds/$DATASET/1.0.0/instances-shape_label.labels.txt" \
      "gs://kubric-public/tfds/$DATASET/1.0.0/instances-size_label.labels.txt" \
      "gs://kubric-public/tfds/$DATASET/1.0.0/movi_a-$SPLIT*" \
      "$DATASET_DIR/"
  fi
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "Error: local TFDS dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

if ! ls "$DATASET_DIR"/movi_a-"$SPLIT".tfrecord-* >/dev/null 2>&1; then
  echo "Error: no MOVi-A $SPLIT TFRecord shards found in: $DATASET_DIR" >&2
  exit 1
fi

cmd=(
  conda run -n "$ENV_NAME"
  python "$SCRIPT_DIR/annotation_utils/convert_kubric_movi.py"
  --data_dir "$DATA_DIR"
  --dataset "$DATASET"
  --split "$SPLIT"
  --output_root "$OUTPUT_ROOT"
  --touch_radius "$TOUCH_RADIUS"
  --no_touch_fps "$NO_TOUCH_FPS"
  --hard_negative_window "$HARD_NEGATIVE_WINDOW"
  --video_fps "$VIDEO_FPS"
)

if [ -n "$MAX_VIDEOS" ]; then
  cmd+=(--max_videos "$MAX_VIDEOS")
fi

echo "Running converter..."
"${cmd[@]}"

echo "Done."
echo "Annotations: $OUTPUT_ROOT/annotations"
