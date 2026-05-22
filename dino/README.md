# DINO Touch Classifier

## Environments

Use the macOS environment locally:

```bash
conda env create -f touch_from_segmentation/dino/environment-mac.yml
conda activate dino-touch-mac
```

Use the Snellius/Linux GPU environment on the cluster:

```bash
conda env create -f touch_from_segmentation/dino/environment-snellius.yml
conda activate dino-touch-snellius
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The Snellius file uses CUDA 12.4 through the `pytorch` and `nvidia` conda channels.
If your Snellius partition or module setup expects another CUDA/PyTorch version,
adjust `pytorch-cuda=12.4` before creating the environment.

## Model Access

The default model is `facebook/dinov3-vits16-pretrain-lvd1689m`. This Hugging Face
checkpoint is gated, so you must request access before the first download.

After access is granted, either log in once:

```bash
huggingface-cli login
```

or pass a token directly:

```bash
python -m touch_from_segmentation.dino.inspect_processor \
  --image touch_from_segmentation/data/kubric_movi_a_256_test/frames/3835/frame_000000.jpg \
  --hf-token YOUR_TOKEN
```

## Inspect Processor Input

To see what the Hugging Face image processor feeds into DINO for a single image:

```bash
python -m touch_from_segmentation.dino.inspect_processor \
  --image touch_from_segmentation/data/kubric_movi_a_256_test/frames/3835/frame_000000.jpg \
  --preview-out checkpoints/dino/processor_preview.png \
  --report-out checkpoints/dino/processor_report.json
```

The command prints and optionally saves the original image size, processor settings,
`pixel_values` tensor shape/statistics, and a de-normalized preview image.

## Snellius Smoke Train

Backslashes must be the final character on each continued line; do not put spaces
after them.

```bash
python -m touch_from_segmentation.dino.main train \
  --annotation-root touch_from_segmentation/data/epic_kitchen/annotations \
  --hf-token YOUR_TOKEN \
  --output-dir touch_from_segmentation/checkpoints/dino_smoke \
  --epochs 1 \
  --batch-size 4 \
  --max-samples 20
```

Add wandb tracking:

```bash
python -m touch_from_segmentation.dino.main train \
  --annotation-root touch_from_segmentation/data/epic_kitchen/annotations \
  --hf-token YOUR_TOKEN \
  --output-dir touch_from_segmentation/checkpoints/dino_epic_mlp \
  --epochs 20 \
  --batch-size 32 \
  --head-type mlp \
  --mlp-layers 2 \
  --wandb \
  --wandb-project dino-touch \
  --wandb-name epic-mlp
```

Use `--wandb-mode offline` on the cluster if outbound network access is unavailable.
The saved `metrics.json` includes a `hyperparameters` object with all CLI options
except the raw Hugging Face token; it records only whether a token was provided.
