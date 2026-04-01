#!/usr/bin/env python3
"""
ast_dataset_make_new.py  —  Cow-independent 5-fold CV dataset split.

Strategy
--------
1. Fixed TEST set   : 5 pre-defined cows (3205, 3214, 3228, 3240, 3277).
                      test.json = all burp clips from those cows
                               + 2× random sample of their nonburp clips.

2. Remaining cows   : Assigned to 5 folds using a *greedy balanced bin-packing*
                      algorithm (LPT heuristic) so each fold holds ~1/5 of the
                      total burp clips, keeping the cow-group intact.

3. Per fold i       : val  = cows in bucket[i]   — all their burp clips
                              + 2× random sample of their nonburp clips
                      train = cows in remaining 4 buckets — same sampling logic.

4. Output           : <output_dir>/
                          train_fold{i}.json
                          val_fold{i}.json
                          test.json
                          label_index.csv

JSON format (identical to original ast_dataset_make.py):
    {"data": [{"wav": "/abs/path/clip.WAV", "labels": "/m/POS"}, ...]}

Usage
-----
    python dataset_make.py \
        --clips_dir /path/to/clips \
        --output_dir /path/to/output \
        --n_folds 5 \
        --nonburp_ratio 2.0 \
        --seed 2026
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


class _Tee:
    """Duplicates writes to two streams (e.g. stdout + a log file)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_COWS = {"3205", "3214", "3228", "3240", "3277"}

DICT_LBS = {
    "burp":    "/m/POS",
    "nonburp": "/m/NEG",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_cow_id(filename: str) -> str:
    """Extract cowID from clip filename.

    Filename format: {cowID}_Round_{roundID}_...WAV
    cowID is everything before the first underscore that is followed by 'Round'.
    In practice it is simply the first token split on '_Round_'.
    """
    basename = os.path.basename(filename)
    return basename.split("_Round_")[0]


def gather_clips(clips_dir: str, class_name: str) -> Dict[str, List[str]]:
    """Return {cowID: [abs_path, ...]} for all clips in clips_dir/<class_name>/."""
    folder = os.path.join(clips_dir, class_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
    cow_clips: Dict[str, List[str]] = defaultdict(list)
    for fname in sorted(os.listdir(folder)):          # sorted → deterministic order
        if not fname.upper().endswith(".WAV"):
            continue
        cow_id = parse_cow_id(fname)
        cow_clips[cow_id].append(os.path.join(folder, fname))
    return dict(cow_clips)


def greedy_balanced_fold_assign(
    cow_burp_counts: Dict[str, int],
    n_folds: int,
) -> Dict[str, int]:
    """Assign each cow to a fold using greedy LPT bin-packing.

    Cows are processed largest-first; each is placed in the fold that
    currently has the fewest total burp clips.  This minimises imbalance.

    Returns {cow_id: fold_index}.
    """
    sorted_cows = sorted(cow_burp_counts, key=cow_burp_counts.get, reverse=True)
    fold_totals = [0] * n_folds
    assignment: Dict[str, int] = {}
    for cow in sorted_cows:
        target_fold = fold_totals.index(min(fold_totals))
        assignment[cow] = target_fold
        fold_totals[target_fold] += cow_burp_counts[cow]
    return assignment


def sample_nonburp(
    cow_nonburp: Dict[str, List[str]],
    target_cows: List[str],
    n_burps: int,
    ratio: float,
    rng: random.Random,
) -> List[str]:
    """Random-sample up to ratio×n_burps nonburp clips from target_cows' pool."""
    pool: List[str] = []
    for cow in target_cows:
        pool.extend(cow_nonburp.get(cow, []))
    rng.shuffle(pool)
    want = int(math.floor(ratio * n_burps))
    if len(pool) < want:
        print(
            f"  [WARNING] nonburp pool has {len(pool)} clips "
            f"but {want} requested (using all available)."
        )
        return pool
    return pool[:want]


def build_json_data(
    burp_paths: List[str],
    nonburp_paths: List[str],
) -> dict:
    data = []
    for p in burp_paths:
        data.append({"wav": p, "labels": DICT_LBS["burp"]})
    for p in nonburp_paths:
        data.append({"wav": p, "labels": DICT_LBS["nonburp"]})
    return {"data": data}


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    # ------------------------------------------------------------------ #
    # 0. Prepare output directory
    # ------------------------------------------------------------------ #
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tee stdout → screen + log file
    log_path = os.path.join(args.output_dir, "split_log.txt")
    _log_file = open(log_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_file)
    try:
        _main_body(args, rng)
    finally:
        sys.stdout = _orig_stdout
        _log_file.close()
    print(f"[INFO] Split log saved → {log_path}")


def _main_body(args: argparse.Namespace, rng: random.Random) -> None:

    # ------------------------------------------------------------------ #
    # 1. Gather all clips, keyed by cowID
    # ------------------------------------------------------------------ #
    cow_burp   = gather_clips(args.clips_dir, "burp")
    cow_nonburp = gather_clips(args.clips_dir, "nonburp")

    all_cows = sorted(set(list(cow_burp.keys()) + list(cow_nonburp.keys())))
    print(f"[INFO] Total cows found      : {len(all_cows)}")
    print(f"[INFO] Test cows (fixed)     : {sorted(TEST_COWS)}")

    # ------------------------------------------------------------------ #
    # 2. Build TEST set
    # ------------------------------------------------------------------ #
    test_burp_paths: List[str] = []
    for cow in sorted(TEST_COWS):
        test_burp_paths.extend(cow_burp.get(cow, []))
    rng.shuffle(test_burp_paths)        # shuffle for good measure

    test_nonburp_paths = sample_nonburp(
        cow_nonburp, sorted(TEST_COWS),
        len(test_burp_paths), args.nonburp_ratio, rng,
    )

    test_json_path = os.path.join(args.output_dir, "test.json")
    save_json(build_json_data(test_burp_paths, test_nonburp_paths), test_json_path)
    print(
        f"\n[TEST] burp={len(test_burp_paths)}, "
        f"nonburp={len(test_nonburp_paths)} → {test_json_path}"
    )

    # ------------------------------------------------------------------ #
    # 3. Remaining cows → balanced 5-fold assignment
    # ------------------------------------------------------------------ #
    remaining_cows = [c for c in all_cows if c not in TEST_COWS]
    print(f"\n[INFO] Remaining cows for CV : {len(remaining_cows)}")

    # Count burp clips per remaining cow (cows with 0 burp get fold assignment too)
    cow_burp_count = {cow: len(cow_burp.get(cow, [])) for cow in remaining_cows}
    total_remaining_burps = sum(cow_burp_count.values())

    fold_assignment = greedy_balanced_fold_assign(cow_burp_count, args.n_folds)

    # Print fold composition
    print(f"\n[INFO] Greedy fold assignment ({args.n_folds} folds):")
    fold_cows: Dict[int, List[str]] = defaultdict(list)
    for cow, fold_idx in fold_assignment.items():
        fold_cows[fold_idx].append(cow)
    for fi in range(args.n_folds):
        cows_in_fold = fold_cows[fi]
        burps_in_fold = sum(cow_burp_count[c] for c in cows_in_fold)
        print(
            f"  Fold {fi}: cows={sorted(cows_in_fold)}, "
            f"burps={burps_in_fold} ({burps_in_fold/total_remaining_burps*100:.1f}%)"
        )

    # ------------------------------------------------------------------ #
    # 4. Generate train/val JSONs for each fold
    # ------------------------------------------------------------------ #
    print()
    for fi in range(args.n_folds):
        val_cows   = fold_cows[fi]
        train_cows = [c for c in remaining_cows if fold_assignment[c] != fi]

        # --- Val ---
        val_burp_paths: List[str] = []
        for cow in val_cows:
            val_burp_paths.extend(cow_burp.get(cow, []))
        rng.shuffle(val_burp_paths)

        val_nonburp_paths = sample_nonburp(
            cow_nonburp, val_cows,
            len(val_burp_paths), args.nonburp_ratio, rng,
        )

        # --- Train ---
        train_burp_paths: List[str] = []
        for cow in train_cows:
            train_burp_paths.extend(cow_burp.get(cow, []))
        rng.shuffle(train_burp_paths)

        train_nonburp_paths = sample_nonburp(
            cow_nonburp, train_cows,
            len(train_burp_paths), args.nonburp_ratio, rng,
        )

        # --- Save ---
        train_path = os.path.join(args.output_dir, f"train_fold{fi}.json")
        val_path   = os.path.join(args.output_dir, f"val_fold{fi}.json")
        save_json(build_json_data(train_burp_paths, train_nonburp_paths), train_path)
        save_json(build_json_data(val_burp_paths,   val_nonburp_paths),   val_path)

        n_train_total = len(train_burp_paths) + len(train_nonburp_paths)
        n_val_total   = len(val_burp_paths)   + len(val_nonburp_paths)
        ratio_str = (
            f"{len(train_burp_paths)/len(val_burp_paths):.2f}:1"
            if val_burp_paths else "N/A"
        )
        print(
            f"[Fold {fi}] "
            f"train burp={len(train_burp_paths)}, nonburp={len(train_nonburp_paths)} "
            f"(total={n_train_total}) | "
            f"val burp={len(val_burp_paths)}, nonburp={len(val_nonburp_paths)} "
            f"(total={n_val_total}) | "
            f"train/val burp ratio={ratio_str}"
        )
        print(f"  → {train_path}")
        print(f"  → {val_path}")

    # ------------------------------------------------------------------ #
    # 5. label_index.csv
    # ------------------------------------------------------------------ #
    label_index_path = os.path.join(args.output_dir, "label_index.csv")
    with open(label_index_path, "w", encoding="utf-8") as f:
        f.write("index,mid,display_name\n")
        for idx, (cls_name, mid) in enumerate(DICT_LBS.items()):
            f.write(f"{idx},{mid},{cls_name}\n")
    print(f"\n[INFO] label_index.csv → {label_index_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cow-independent 5-fold CV dataset split with balanced bin-packing."
    )
    parser.add_argument(
        "--clips_dir", required=True,
        help="Root clips directory containing 'burp/' and 'nonburp/' sub-folders.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for JSON files.",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--nonburp_ratio", type=float, default=2.0,
        help="Nonburp-to-burp sampling ratio (default: 2.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed for reproducibility (default: 2026).",
    )
    main(parser.parse_args())
