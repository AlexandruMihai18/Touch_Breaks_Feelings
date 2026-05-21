"""Plot the spatial distribution of touch events across an N×N grid.

Reads one or more annotation JSON files (which must already have x_touch/y_touch
populated by annotate_touch_zones.py) and renders a heatmap of touch-event counts
per grid cell.  No model predictions involved.

Usage
-----
    python scripts/visualize_touch_zones.py \\
        data/greatest_hits/annotations/train.json \\
        data/epic_kitchen/annotations/train.json \\
        [--grid 8] [--output plots/touch_zone_distribution.png] [--title "My dataset"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def count_zones(annotation_paths: list[Path], grid: int) -> np.ndarray:
    counts = np.zeros((grid, grid), dtype=int)
    for path in annotation_paths:
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            x, y = e.get("x_touch"), e.get("y_touch")
            if x is None or y is None:
                continue
            xi, yi = int(x), int(y)
            if 0 <= xi < grid and 0 <= yi < grid:
                counts[yi, xi] += 1
    return counts


def plot_zone_distribution(
    counts: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    grid = counts.shape[0]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(counts, cmap="YlOrRd", aspect="equal", vmin=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Touch event count", fontsize=11)

    for y in range(grid):
        for x in range(grid):
            n = counts[y, x]
            pct = 100 * n / total if total > 0 else 0
            cell_txt = f"{n}\n{pct:.1f}%"
            text_color = "white" if n > counts.max() * 0.6 else "black"
            ax.text(x, y, cell_txt, ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))
    ax.set_xticklabels([str(i) for i in range(grid)])
    ax.set_yticklabels([str(i) for i in range(grid)])
    ax.set_xlabel("x zone  (left → right)", fontsize=11)
    ax.set_ylabel("y zone  (top → bottom)", fontsize=11)
    ax.set_title(f"{title}\n(n={total} touch events)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="+", type=Path,
                        help="Annotation JSON files with x_touch/y_touch already set")
    parser.add_argument("--grid", type=int, default=8, metavar="N",
                        help="Grid dimension (default: 8)")
    parser.add_argument("--title", type=str, default="Touch zone distribution",
                        help="Plot title (default: 'Touch zone distribution')")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: touch_zone_distribution.png next to first annotation file)")
    args = parser.parse_args()

    missing = [p for p in args.annotations if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Not found: {missing}")

    counts = count_zones(args.annotations, args.grid)
    output = args.output or args.annotations[0].parent / "touch_zone_distribution.png"
    plot_zone_distribution(counts, args.title, output)
    print(f"Saved → {output}  (total touch events: {counts.sum()})")


if __name__ == "__main__":
    main()
