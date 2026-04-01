#!/usr/bin/env python3
"""
Generate audio clips from raw audio files based on annotation metadata.

Usage:
    python generate_clips.py \
        --metadata annotation_for_binary_classification.csv \
        --audio_folder /path/to/raw_audio \
        --output_folder /path/to/output \
        [--seed 2026]

Directory structure assumed:
    <audio_folder>/
        <cowID>/
            Round_1_202405101500_05141100/
            Round_2_202405141100_05171100/
            Round_3_202405171100_05211000/

Output structure:
    <output_folder>/
        burp/
            <cowID>_Round_<roundID>_<3digits>_<start>_<end>.WAV
        nonburp/
            <cowID>_Round_<roundID>_<3digits>_<start>_<end>.WAV
"""

import argparse
import os
import random
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_FOLDER_MAP = {
    1: "Round_1_202405101500_05141100",
    2: "Round_2_202405141100_05171100",
    3: "Round_3_202405171100_05211000",
}

TARGET_DURATION = 5.0  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_audio_file(audio_folder: str, cow_id: str, round_id: int, raw_name: str) -> Optional[str]:
    """Return the absolute path to the raw audio WAV file, or None if not found."""
    round_folder = ROUND_FOLDER_MAP.get(round_id)
    if round_folder is None:
        return None
    candidate = os.path.join(audio_folder, str(cow_id), round_folder, raw_name)
    if os.path.isfile(candidate):
        return candidate
    # Case-insensitive fallback (useful on case-sensitive Linux filesystems
    # where the annotation might differ in capitalisation)
    parent = os.path.join(audio_folder, str(cow_id), round_folder)
    if os.path.isdir(parent):
        raw_name_lower = raw_name.lower()
        for fname in os.listdir(parent):
            if fname.lower() == raw_name_lower:
                return os.path.join(parent, fname)
    return None


def get_three_digit_suffix(raw_name: str) -> str:
    """Extract the last 3 characters before the file extension.

    E.g. 'Trial_1_2K4_Round_1_202405101500_05141100_025.WAV' -> '025'
    """
    stem = os.path.splitext(raw_name)[0]  # remove extension
    return stem[-3:]


def pad_burp_clip(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    file_duration_sec: float,
    rng: random.Random,
) -> Tuple[float, float]:
    """Randomly expand [start_sec, end_sec] to TARGET_DURATION seconds.

    The burp is kept centered-ish within the 5-second window but randomised
    so that the left pad length is drawn uniformly from the feasible range.

    Returns the new (clip_start, clip_end) in seconds (floats, not rounded).
    """
    clip_duration = end_sec - start_sec
    pad_total = TARGET_DURATION - clip_duration  # > 0 for burps < 5 s

    # Maximum left pad is constrained by both the total pad and by start of file
    max_left_pad = min(pad_total, start_sec)
    # Right pad must cover the remainder; check it doesn't exceed file end
    # left_pad in [min_left_pad, max_left_pad]
    min_left_pad = max(0.0, pad_total - (file_duration_sec - end_sec))
    min_left_pad = max(min_left_pad, 0.0)

    if min_left_pad > max_left_pad:
        # Edge case: file too short; just clip from 0 or to file end
        left_pad = min_left_pad
    else:
        left_pad = rng.uniform(min_left_pad, max_left_pad)

    new_start = start_sec - left_pad
    new_end = new_start + TARGET_DURATION

    # Clamp to file boundaries (safety)
    new_start = max(0.0, new_start)
    new_end = min(file_duration_sec, new_end)

    return new_start, new_end


def extract_and_save_clip(
    audio: np.ndarray,
    sr: int,
    clip_start: float,
    clip_end: float,
    out_path: str,
) -> None:
    """Slice [clip_start, clip_end] from audio and write to out_path."""
    start_sample = int(round(clip_start * sr))
    end_sample = int(round(clip_end * sr))
    # Clamp
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    clip = audio[start_sample:end_sample]
    sf.write(out_path, clip, sr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 5-second audio clips from raw audio based on annotation CSV."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to annotation_for_binary_classification.csv",
    )
    parser.add_argument(
        "--audio_folder",
        required=True,
        help="Root folder of raw audio (contains sub-folders named by cowID).",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Root output folder. Sub-folders 'burp' and 'nonburp' will be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible burp-clip padding (default: 42).",
    )
    args = parser.parse_args()

    # ---- Setup ----
    rng = random.Random(args.seed)

    burp_dir = os.path.join(args.output_folder, "burp")
    nonburp_dir = os.path.join(args.output_folder, "nonburp")
    os.makedirs(burp_dir, exist_ok=True)
    os.makedirs(nonburp_dir, exist_ok=True)

    # ---- Load metadata ----
    df = pd.read_csv(args.metadata)
    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    required_cols = {"cowID", "roundID", "rawname_arc", "start_time", "end_time", "event"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Filter to only burp / nonburp rows
    df = df[df["event"].isin(["burp", "nonburp"])].reset_index(drop=True)
    print(f"Total annotations to process: {len(df)}")

    # ---- Cache open audio files to avoid repeated reads ----
    audio_cache: Dict[str, Tuple[np.ndarray, int, float]] = {}  # path -> (audio, sr, dur)

    skipped_not_found = 0
    skipped_burp_too_long = 0
    saved = 0
    errors = 0

    for idx, row in df.iterrows():
        cow_id = str(row["cowID"]).strip()
        round_id = int(row["roundID"])
        raw_name = str(row["rawname_arc"]).strip()
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        event = str(row["event"]).strip().lower()

        duration = end_time - start_time

        # ---- Truncate nonburp clips > 5 s to exactly 5 s ----
        if event == "nonburp" and duration > TARGET_DURATION:
            end_time = start_time + TARGET_DURATION

        # Burp clips > 5 s are unexpected but skip them gracefully
        if event == "burp" and duration > TARGET_DURATION:
            skipped_burp_too_long += 1
            print(
                f"  WARNING: burp clip longer than {TARGET_DURATION}s at row {idx+2} "
                f"({raw_name}, {start_time:.3f}-{end_time:.3f}) — skipped."
            )
            continue

        # ---- Locate audio file ----
        audio_path = find_audio_file(args.audio_folder, cow_id, round_id, raw_name)
        if audio_path is None:
            skipped_not_found += 1
            print(
                f"  WARNING: audio file not found for row {idx+2}: "
                f"cowID={cow_id}, roundID={round_id}, file={raw_name}"
            )
            continue

        # ---- Load audio (with cache) ----
        if audio_path not in audio_cache:
            try:
                audio, sr = sf.read(audio_path, always_2d=False)
                file_dur = len(audio) / sr
                audio_cache[audio_path] = (audio, sr, file_dur)
            except Exception as e:
                print(f"  ERROR reading {audio_path}: {e}", file=sys.stderr)
                errors += 1
                continue
        audio, sr, file_dur = audio_cache[audio_path]

        # ---- Determine clip boundaries ----
        if event == "burp":
            clip_start, clip_end = pad_burp_clip(
                audio, sr, start_time, end_time, file_dur, rng
            )
        else:
            clip_start = start_time
            clip_end = end_time

        # ---- Build output filename ----
        three_digits = get_three_digit_suffix(raw_name)
        # Times in filename are rounded integers (no decimal point)
        start_int = int(round(clip_start))
        end_int = int(round(clip_end))
        out_name = f"{cow_id}_Round_{round_id}_{three_digits}_{start_int:04d}_{end_int:04d}.WAV"

        out_dir = burp_dir if event == "burp" else nonburp_dir
        out_path = os.path.join(out_dir, out_name)

        # Skip if already exists (idempotent re-run)
        if os.path.exists(out_path):
            saved += 1  # count as done
            continue

        # ---- Extract and save ----
        try:
            extract_and_save_clip(audio, sr, clip_start, clip_end, out_path)
            saved += 1
        except Exception as e:
            print(f"  ERROR writing {out_path}: {e}", file=sys.stderr)
            errors += 1

    print("\n=== Summary ===")
    print(f"  Clips saved/already exist : {saved}")
    print(f"  Skipped (audio not found) : {skipped_not_found}")
    print(f"  Skipped (burp   > 5s)     : {skipped_burp_too_long}")
    print(f"  Errors                    : {errors}")


if __name__ == "__main__":
    main()
