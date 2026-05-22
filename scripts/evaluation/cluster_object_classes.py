"""Cluster EPIC-Kitchen object labels into semantic groups for evaluation.

Reads object_labels.txt (global contact-frame counts) and keeps only objects
with count >= MIN_COUNT. Then applies two parallel clusterings driven by JSON
rule files:

  1. Seven semantic clusters suited to kitchen-scene evaluation.
     Rules file: cluster_rules_7.json  (alongside object_labels.txt)
  2. Greatest-Hits-compatible clusters (Ceramic / Cloth / Glass / Plastic bag /
     Water / Wood) to allow direct comparison with the GH benchmark.
     Rules file: cluster_rules_gh.json

Each rules JSON has the schema:
    {
      "explicit":      { "object_name": "ClusterLabel", ... },
      "keyword_rules": [["keyword", "ClusterLabel"], ...]
    }

Explicit entries take priority; keyword rules are tried in order (word-boundary
match); anything unmatched falls through to "Other".

Outputs two JSON files: {output_dir}/object_clusters_7.json and
{output_dir}/object_clusters_gh.json, both with the schema:
    { "object_name": "ClusterLabel", ... }

Objects with no GH mapping are omitted from the GH file.

Usage
-----
    python scripts/evaluation/cluster_object_classes.py \\
        data/epic_kitchen/object_labels.txt \\
        [--min-count 1] \\
        [--rules-dir data/epic_kitchen/] \\
        [--output-dir data/epic_kitchen/]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

GH_CATEGORIES     = ["Ceramic", "Cloth", "Glass", "Plastic bag", "Water", "Wood"]
SEVEN_CATEGORIES  = ["Utensils", "Cookware", "Tableware", "Textiles", "Food", "Appliances", "Packaging"]


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------

def load_rules(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    with open(path) as f:
        data = json.load(f)
    explicit = data["explicit"]
    keyword_rules = [tuple(pair) for pair in data["keyword_rules"]]
    return explicit, keyword_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign(name: str, explicit: dict[str, str], keyword_rules: list[tuple[str, str]]) -> str:
    if name in explicit:
        return explicit[name]
    lower = name.lower()
    for keyword, cluster in keyword_rules:
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            return cluster
    return "Other"


def parse_labels(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    counts[parts[1].strip()] = int(parts[0])
                except ValueError:
                    pass
    return counts


def build_mapping(
    counts: dict[str, int],
    min_count: int,
    explicit_7: dict[str, str],
    rules_7: list[tuple[str, str]],
    explicit_gh: dict[str, str],
    rules_gh: list[tuple[str, str]],
) -> tuple[dict, dict]:
    filtered = {name: cnt for name, cnt in counts.items() if cnt >= min_count}
    mapping_7  = {name: _assign(name, explicit_7, rules_7) for name in filtered}
    mapping_gh = {
        name: label
        for name in filtered
        if (label := _assign(name, explicit_gh, rules_gh)) != "Other"
    }
    return mapping_7, mapping_gh


def _print_cluster_table(mapping: dict[str, str], counts: dict[str, int], categories: list[str]) -> None:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, cluster in mapping.items():
        grouped[cluster].append((name, counts.get(name, 0)))

    total_samples = sum(counts.get(n, 0) for n in mapping)
    print(f"{'Cluster':<16}  {'Objects':>7}  {'Samples':>8}  {'Share':>6}  Top objects")
    print("-" * 90)
    for cat in categories + (["Other"] if "Other" in grouped else []):
        items = sorted(grouped.get(cat, []), key=lambda x: -x[1])
        n_items    = len(items)
        n_samples  = sum(c for _, c in items)
        share      = n_samples / total_samples * 100 if total_samples else 0
        top        = ", ".join(f"{n}({c})" for n, c in items[:4])
        print(f"{cat:<16}  {n_items:>7}  {n_samples:>8}  {share:>5.1f}%  {top}")
    print(f"\n  Total: {len(mapping)} objects, {total_samples:,} samples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="object_labels.txt from data/epic_kitchen/")
    parser.add_argument("--min-count", type=int, default=1, metavar="N",
                        help="Minimum global sample count to include a class (default: 1)")
    parser.add_argument("--rules-dir", type=Path, default=None,
                        help="Directory containing cluster_rules_7.json and cluster_rules_gh.json "
                             "(default: same directory as labels file)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for output JSON files (default: same dir as labels file)")
    args = parser.parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(args.labels)

    rules_dir = args.rules_dir or args.labels.parent
    explicit_7,  rules_7  = load_rules(rules_dir / "cluster_rules_7.json")
    explicit_gh, rules_gh = load_rules(rules_dir / "cluster_rules_gh.json")

    counts = parse_labels(args.labels)
    mapping_7, mapping_gh = build_mapping(
        counts, args.min_count, explicit_7, rules_7, explicit_gh, rules_gh
    )

    out_dir = args.output_dir or args.labels.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    path_7  = out_dir / "object_clusters_7.json"
    path_gh = out_dir / "object_clusters_gh.json"

    with open(path_7, "w") as f:
        json.dump(mapping_7, f, indent=2, ensure_ascii=False)
    with open(path_gh, "w") as f:
        json.dump(mapping_gh, f, indent=2, ensure_ascii=False)

    print(f"\n=== 7-cluster mapping  (min_count={args.min_count}) ===\n")
    _print_cluster_table(mapping_7, counts, SEVEN_CATEGORIES)

    print(f"\n=== Greatest Hits compatible mapping  (Ceramic / Cloth / Glass / Plastic bag / Water / Wood) ===\n")
    _print_cluster_table(mapping_gh, counts, GH_CATEGORIES)
    unmapped = len(mapping_7) - len(mapping_gh)
    print(f"  {unmapped} objects have no GH-compatible mapping (kitchen-specific: tools, food, appliances, etc.)")

    print(f"\nSaved → {path_7}")
    print(f"Saved → {path_gh}")


if __name__ == "__main__":
    main()
