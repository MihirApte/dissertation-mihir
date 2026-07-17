"""
clean_results.py
=================
Copies every result GIF out of the messy results/ tree into a flat,
cleanly-named results_clean/ folder - no config.yaml files, no nested
date/prompt/index subfolders, no long parameter-suffix filenames.

Original results/ folder (GIFs and config.yaml) is never touched or
deleted - this only ever copies, so it's safe to re-run anytime, including
partway through a long experiment run, as a progress check.

Output filenames look like:
    truck_baseline_random_Wooden-trucks-drive-on-a-racetrack.gif

USAGE (run from inside the RAVE project folder):
    python3 clean_results.py
"""

import os
import re
import shutil
import yaml

SOURCE_DIR = "results"
OUT_DIR = "results_clean"

# Same keyword set compute_metrics_all.py and compare_gifs.py already use -
# keeping this list in sync with those means clean_results.py recognises
# exactly the same set of methods.
METHOD_KEYWORDS = [
    "semantic_multicontrol",   # check this before multicontrol_random/semantic_shuffle
    "baseline_random",
    "semantic_shuffle",
    "kmeans_shuffle",
    "multicontrol_random",
]


def slugify(text):
    text = text.strip()
    text = re.sub(r"[^\w\s-]", "", text)   # drop punctuation (apostrophes, commas, ...)
    text = re.sub(r"\s+", "-", text)       # spaces -> dashes
    return text


def find_method(path_and_folder):
    for kw in METHOD_KEYWORDS:
        if kw in path_and_folder:
            return kw
    return None


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"'{SOURCE_DIR}/' not found - nothing to clean yet.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    copied, skipped_dupe, warned = 0, 0, 0

    for dirpath, _dirs, files in os.walk(SOURCE_DIR):
        if "config.yaml" not in files:
            continue

        gif_files = [f for f in files if f.endswith(".gif")]
        if len(gif_files) == 0:
            print(f"[WARN] no .gif found next to config.yaml in {dirpath}")
            warned += 1
            continue
        if len(gif_files) > 1:
            print(f"[WARN] multiple .gif files in {dirpath}, using the first one")
            warned += 1

        cfg_path = os.path.join(dirpath, "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        video = cfg.get("video_name", "unknown")
        prompt = cfg.get("positive_prompts", "")
        method = find_method(cfg.get("save_folder", "") + dirpath)

        if method is None:
            print(f"[WARN] could not identify method for {dirpath} - skipping")
            warned += 1
            continue

        target_name = f"{video}_{method}_{slugify(prompt)}.gif"
        target_path = os.path.join(OUT_DIR, target_name)

        if os.path.exists(target_path):
            print(f"[SKIP] {target_name} - already in {OUT_DIR}/ (duplicate run)")
            skipped_dupe += 1
            continue

        src_gif = os.path.join(dirpath, gif_files[0])
        shutil.copy2(src_gif, target_path)
        print(f"[COPY] {target_name}")
        copied += 1

    print(f"\nDone. {copied} GIF(s) copied to {OUT_DIR}/, "
          f"{skipped_dupe} duplicate(s) skipped, {warned} warning(s).")


if __name__ == "__main__":
    main()
