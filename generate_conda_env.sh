#!/usr/bin/env bash
set -euo pipefail

# Repository dir
REPO_DIR="${1:-$(pwd)}"

# Prefer using the env's python directly (avoid relying on activation)
PY_BIN="$REPO_DIR/.venv/bin/python"
ACTIVATE="$REPO_DIR/.venv/bin/activate"
ACTIVATED=0

if [ ! -x "$PY_BIN" ]; then
  if [ -f "$ACTIVATE" ]; then
    echo "source $ACTIVATE"
    # shellcheck source=/dev/null
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    ACTIVATED=1
    PY_BIN="$(which python)"
  else
    echo "Error: Python not found in virtualenv and activate script missing."
    exit 1
  fi
fi

# Read project name from pyproject.toml (fallback if not found)
PROJECT_NAME=$(grep -E '^\s*name\s*=' "$REPO_DIR/pyproject.toml" \
  | head -n1 \
  | sed -E 's/.*=\s*"([^"]+)".*/\1/' || true)
PROJECT_NAME="${PROJECT_NAME:-myenv}"

# Run sync env from $REPO_DIR
echo "cd $REPO_DIR && uv sync"
cd "$REPO_DIR" || exit
uv sync

# Determine Python major.minor.patch version from the env
echo "$PY_BIN -c 'import sys; print(\"{}.{}.{}\".format(*sys.version_info[:3]))'"
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Get pip freeze output (keeps VCS and editable lines)
echo "uv pip list --python $PY_BIN --format freeze"
PIP_FREEZE=$(uv pip list --python "$PY_BIN" --format freeze)

# Remove packages that conflict or are managed by torch itself:
# - torch / torchvision / torchaudio: re-added explicitly with CUDA index
# - triton: pinned by torch; let pip resolve it
# - nvidia-*: CUDA runtime libs bundled with torch wheels; let pip resolve them
# - editable installs (-e ...): not portable
# - the project itself
PROJECT_REGEX="^${PROJECT_NAME//_/[-_]}=="
PIP_FREEZE=$(echo "$PIP_FREEZE" \
  | grep -v '^torch==' \
  | grep -v '^torchvision==' \
  | grep -v '^torchaudio==' \
  | grep -v '^triton==' \
  | grep -vi '^nvidia-' \
  | grep -v '^-e ' \
  | grep -Ev "$PROJECT_REGEX" || true)

CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"

# Write conda YAML
{
  echo "name: $PROJECT_NAME"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  echo "  - ffmpeg"
  echo "  - pip:"
  # ---- PyTorch (CUDA 12.4) ----
  echo "    - --extra-index-url https://download.pytorch.org/whl/cu124"
  echo '    - "torch==2.5.1"'
  echo '    - "torchvision==0.20.1"'

  if [ -z "$PIP_FREEZE" ]; then
    echo "    # no additional pip packages detected"
  else
    while IFS= read -r pkg; do
      [ -z "$pkg" ] && continue
      printf '    - "%s"\n' "$(printf '%s' "$pkg" | sed 's/"/\\"/g')"
    done <<< "$PIP_FREEZE"
  fi
} > "$CONDA_YAML_PATH"

echo "Generated conda environment YAML at: $CONDA_YAML_PATH"

if [ "$ACTIVATED" -eq 1 ]; then
  echo "deactivate"
  deactivate || true
fi