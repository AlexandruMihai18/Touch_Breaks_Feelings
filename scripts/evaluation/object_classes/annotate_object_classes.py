"""Annotate each sample with semantic object-class labels derived from its
``object_name`` field.

Two fields are written per entry:

  general_class  – one of the 7 semantic kitchen clusters produced by
                   cluster_object_classes.py (Utensils / Cookware / Tableware /
                   Textiles / Food / Appliances / Packaging / Other).
                   Set to null when object_name is absent or empty.

  gh_class       – Greatest-Hits-compatible class (Ceramic / Cloth / Glass /
                   Plastic bag / Water / Wood) for direct comparison with the
                   GH benchmark.  Set to null when the object has no GH mapping
                   (kitchen-specific items) or when object_name is absent.

Cluster mappings are pre-built JSON files produced by
``cluster_object_classes.py``:
  object_clusters_7.json   → {object_name: general_class, ...}
  object_clusters_gh.json  → {object_name: gh_class, ...}  (only mapped items)

Usage
-----
    python scripts/evaluation/object_classes/annotate_object_classes.py \\
        data/epic_kitchen/annotations/train.json \\
        --clusters-7  data/epic_kitchen/object_clusters_7.json \\
        --clusters-gh data/epic_kitchen/object_clusters_gh.json \\
        [--force] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def annotate_file(
    path: Path,
    clusters_7: dict[str, str],
    clusters_gh: dict[str, str],
    force: bool,
    dry_run: bool,
) -> dict:
    with open(path) as f:
        annotations = json.load(f)

    counts = {"updated": 0, "skipped": 0, "already_done": 0}

    for entry in annotations:
        if not force and "general_class" in entry and "gh_class" in entry:
            counts["already_done"] += 1
            continue

        name = entry.get("object_name", "").strip()
        if not name:
            entry["general_class"] = None
            entry["gh_class"] = None
            counts["skipped"] += 1
            continue

        entry["general_class"] = clusters_7.get(name, "Other")
        entry["gh_class"] = clusters_gh.get(name)  # None when no GH mapping
        counts["updated"] += 1

    if not dry_run:
        with open(path, "w") as f:
            json.dump(annotations, f, indent=2)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument(
        "--clusters-7", type=Path, required=True, metavar="JSON",
        help="object_clusters_7.json produced by cluster_object_classes.py",
    )
    parser.add_argument(
        "--clusters-gh", type=Path, required=True, metavar="JSON",
        help="object_clusters_gh.json produced by cluster_object_classes.py",
    )
    parser.add_argument("--force",   action="store_true",
                        help="Re-compute even if general_class/gh_class already set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write changes")
    args = parser.parse_args()

    clusters_7  = json.loads(args.clusters_7.read_text())
    clusters_gh = json.loads(args.clusters_gh.read_text())

    for path in args.annotations:
        if not path.exists():
            print(f"[WARN] not found: {path}")
            continue
        counts = annotate_file(path, clusters_7, clusters_gh, args.force, args.dry_run)
        tag = " (dry run)" if args.dry_run else ""
        print(
            f"{path}{tag}\n"
            f"  updated={counts['updated']}  skipped={counts['skipped']}  "
            f"already_done={counts['already_done']}"
        )


if __name__ == "__main__":
    main()
