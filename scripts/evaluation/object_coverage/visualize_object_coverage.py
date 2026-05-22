"""Plot the distribution of object mask coverage, split by touch / no-touch.

Reads annotation JSON files with `object_coverage` already populated by
annotate_object_coverage.py and renders overlaid histograms for touch-positive
and no-touch samples.

Usage
-----
    python scripts/evaluation/object_coverage/visualize_object_coverage.py \\
        data/greatest_hits/annotations/train.json \\
        data/epic_kitchen/annotations/train.json \\
        [--bins 20] [--output plots/coverage_distribution.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def collect(annotation_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    touch, no_touch = [], []
    for path in annotation_paths:
        with open(path) as f:
            entries = json.load(f)
        for e in entries:
            c = e.get("object_coverage")
            if c is None:
                continue
            (touch if e.get("type") == "touch" else no_touch).append(float(c))
    return np.array(touch), np.array(no_touch)


def plot(touch: np.ndarray, no_touch: np.ndarray, bins: int, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))

    shared_bins = np.linspace(0, max(touch.max() if len(touch) else 0,
                                     no_touch.max() if len(no_touch) else 0,
                                     0.01), bins + 1)

    if len(touch):
        ax.hist(touch,    bins=shared_bins, alpha=0.65, color="#e05252", label=f"touch (n={len(touch)})",    edgecolor="white", linewidth=0.4)
    if len(no_touch):
        ax.hist(no_touch, bins=shared_bins, alpha=0.65, color="#5281e0", label=f"no-touch (n={len(no_touch)})", edgecolor="white", linewidth=0.4)

    for arr, color in [(touch, "#e05252"), (no_touch, "#5281e0")]:
        if len(arr):
            ax.axvline(np.median(arr), color=color, linestyle="--", linewidth=1.2)

    ax.set_xlabel("Object mask coverage  (nonzero pixels / total pixels)", fontsize=10)
    ax.set_ylabel("Number of samples", fontsize=10)
    ax.set_title(f"{title}\n(n={len(touch) + len(no_touch)} total)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--bins",   type=int, default=25,    help="Number of histogram bins (default: 25)")
    parser.add_argument("--title",  type=str, default="Object mask coverage distribution")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    touch, no_touch = collect(args.annotations)
    if not len(touch) and not len(no_touch):
        print("No object_coverage values found — run annotate_object_coverage.py first.")
        return

    output = args.output or args.annotations[0].parent / "object_coverage_distribution.png"
    plot(touch, no_touch, args.bins, args.title, output)
    print(f"Saved → {output}")
    if len(touch):
        print(f"  touch    median={np.median(touch):.4f}  range={touch.min():.4f}–{touch.max():.4f}")
    if len(no_touch):
        print(f"  no-touch median={np.median(no_touch):.4f}  range={no_touch.min():.4f}–{no_touch.max():.4f}")


if __name__ == "__main__":
    main()
