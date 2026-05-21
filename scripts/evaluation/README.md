# Evaluation Scripts

Post-training analysis tools that characterise model performance along two spatial dimensions:

| Subfolder | Dimension |
| --- | --- |
| `touch_zones/` | Where in the frame (8×8 spatial grid) touch events occur, and model recall per zone |
| `depth/` | How far from the camera (depth proxy) touch events occur, and model recall per depth bin |

---

## Prerequisites

Both subfolders operate on **annotation JSON files** (`train.json` / `val.json`) that must already have been produced by the dataset pipelines (`scripts/greatest_hits/`, `scripts/epic_kitchen/`).

**Required fields on each annotation entry before running evaluation:**

| Field | Added by | Used by |
| --- | --- | --- |
| `image_path`, `target_path` / `touch_mask_path` | dataset pipeline | annotation scripts |
| `x_touch`, `y_touch` | `touch_zones/annotate_touch_zones.py` | `touch_zones/analyze_grid_performance.py` |
| `depth_touch` | `depth/annotate_touch_depth.py` | `depth/analyze_depth_performance.py` |

---

## Typical workflow

```text
[annotation JSONs from dataset pipelines]
        ↓
touch_zones/annotate_touch_zones.py     ← add x_touch, y_touch to each touch entry
depth/annotate_touch_depth.py           ← add depth_touch to each touch entry (GH only)
        ↓
[run model → produce predictions CSV]
        ↓
touch_zones/analyze_grid_performance.py ← spatial heatmap of recall per grid zone
depth/analyze_depth_performance.py      ← bar chart of recall per depth bin
```

The two annotation steps are independent and can run in either order. Both write in-place to the JSON files and are safe to re-run (`--force` to recompute).

---

## Predictions CSV format

The analysis scripts expect a CSV with at least these columns (extra columns are ignored):

```
frame_id, video_id, frame_path, audio_path, label, prediction, x_touch, y_touch, depth_touch
```

`label` and `prediction` are binary integers (0 = no-touch, 1 = touch).  
`x_touch`, `y_touch`, `depth_touch` are populated from the annotation JSONs at eval time; no-touch rows leave them blank/NaN.
