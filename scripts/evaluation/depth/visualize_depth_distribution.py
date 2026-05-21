"""Plot the distribution of median touch depths across annotation files.

Reads annotation JSON files with `depth_touch` already populated by
annotate_touch_depth.py and renders a histogram of depth values.

Usage
-----
    python scripts/visualize_depth_distribution.py \\
        data/greatest_hits/annotations/train.json \\
        [--bins 20] [--output plots/depth_distribution.png] [--title "GH depth"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def collect_depths(annotation_paths: list[Path]) -> np.ndarray:
    depths = []
    for path in annotation_paths:
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            d = e.get("depth_touch")
            if d is not None:
                depths.append(float(d))
    return np.array(depths)


def plot_depth_distribution(depths: np.ndarray, bins: int, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    counts, edges, patches = ax.hist(depths, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)

    # Color bars by depth value (light → dark proxy)
    norm = plt.Normalize(edges[0], edges[-1])
    cmap = plt.cm.YlOrRd
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(cmap(norm(left + (edges[1] - edges[0]) / 2)))

    ax.set_xlabel(
        "Median depth under touch mask\n"
        "(inferno colormap of Depth-Anything-V2 disparity — low = dark/purple = far,  high = bright/yellow = near)",
        fontsize=9,
    )
    ax.set_ylabel("Number of touch events", fontsize=10)
    ax.set_title(f"{title}\n(n={len(depths)} touch samples with depth)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate basic stats
    ax.axvline(np.median(depths), color="black", linestyle="--", linewidth=1.2, label=f"median={np.median(depths):.1f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--bins",   type=int,  default=20,  help="Number of histogram bins (default: 20)")
    parser.add_argument("--title",  type=str,  default="Touch depth distribution")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    depths = collect_depths(args.annotations)
    if len(depths) == 0:
        print("No depth_touch values found — run annotate_touch_depth.py first.")
        return

    output = args.output or args.annotations[0].parent / "depth_distribution.png"
    plot_depth_distribution(depths, args.bins, args.title, output)
    print(f"Saved → {output}  (n={len(depths)}, median={np.median(depths):.1f}, "
          f"range={depths.min():.1f}–{depths.max():.1f})")


if __name__ == "__main__":
    main()
