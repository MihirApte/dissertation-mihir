"""
compute_cost_experiment.py
===========================
Computational-cost comparison: Baseline vs the 3 training-free
improvements (Semantic v1, Semantic v2, Multi-ControlNet + FreeU), on a
small subset of 5 videos (20 runs total).

This is a deliberately SEPARATE, self-contained script -- it does not
touch scripts/run_experiment.py, generate_configs.py, or any of the 86
already-completed dissertation experiments. It only reads the per-method
config YAMLs that already exist in configs/ for these 5 videos, and
drives each one as its own subprocess via compute_cost_single_run.py, so
every measurement starts from a completely clean GPU/Python state (fresh
model load, empty CUDA cache, reset peak-memory counter).

Writes results incrementally to results/computation_cost.txt as each run
finishes, so a crash partway through (e.g. an overnight cluster job that
gets interrupted) still leaves whatever completed so far safely on disk.

USAGE (run from inside the RAVE project folder, on a machine with a GPU):
    python3 compute_cost_experiment.py

Expect this to take a while: 20 runs, each including full model loading,
DDIM inversion, and 50-step denoising -- roughly 20x a single normal
experiment's wall-clock time, since each run reloads everything from
scratch by design (that's what makes the measurements fair/uncached).
"""

import os
import re
import subprocess
import sys

VIDEOS = ["flowers", "motorboat", "skateboarding", "farmland", "waterglass"]

METHODS = [
    ("baseline_random", "Baseline (random, depth_zoe)"),
    ("semantic_shuffle", "Semantic v1 (greedy NN, depth_zoe)"),
    ("kmeans_shuffle", "Semantic v2 (K-means, depth_zoe)"),
    ("multicontrol_random", "Multi-ControlNet (random, depth+canny+FreeU)"),
]

CONFIG_DIR = "configs"
OUT_PATH = "results/computation_cost.txt"

RESULT_RE = re.compile(
    r"RESULT video=(\S+) method=(\S+) wall_time_sec=([\d.]+) "
    r"peak_gpu_mb=([\d.]+) frames=(\d+)"
)


def main():
    os.makedirs("results", exist_ok=True)
    completed = 0

    with open(OUT_PATH, "w") as out:
        out.write("RAVE Dissertation -- Computational Cost Comparison\n\n")
        out.write(
            f"{'Video':<15} {'Method':<44} {'Wall(s)':>10} "
            f"{'PeakGPU(MB)':>12} {'Frames':>7}\n"
        )
        out.write("-" * 92 + "\n")
        out.flush()

        for video in VIDEOS:
            for suffix, label in METHODS:
                config_path = os.path.join(CONFIG_DIR, f"{video}_{suffix}.yaml")

                if not os.path.exists(config_path):
                    print(f"[SKIP] {config_path} not found")
                    out.write(f"{video:<15} {label:<44} {'N/A':>10} {'N/A':>12} {'-':>7}\n")
                    out.flush()
                    continue

                print(f"[RUN]  {video} / {label} ...")
                proc = subprocess.run(
                    [sys.executable, "compute_cost_single_run.py", config_path],
                    capture_output=True,
                    text=True,
                )

                if proc.returncode != 0:
                    print(f"[FAIL] {video} / {label}\n{proc.stderr[-2000:]}")
                    out.write(f"{video:<15} {label:<44} {'FAILED':>10} {'FAILED':>12} {'-':>7}\n")
                    out.flush()
                    continue

                m = RESULT_RE.search(proc.stdout)
                if not m:
                    print(f"[FAIL] {video} / {label} -- could not parse output:\n{proc.stdout}")
                    out.write(f"{video:<15} {label:<44} {'PARSE_ERR':>10} {'PARSE_ERR':>12} {'-':>7}\n")
                    out.flush()
                    continue

                _, _, wall_s, peak_mb, frames = m.groups()
                out.write(
                    f"{video:<15} {label:<44} {float(wall_s):>10.2f} "
                    f"{float(peak_mb):>12.1f} {int(frames):>7}\n"
                )
                out.flush()
                completed += 1
                print(f"[OK]   {video} / {label}: {wall_s}s, {peak_mb} MB, {frames} frames")

            out.write("\n")
            out.flush()

    print(f"\nDone. {completed} of {len(VIDEOS) * len(METHODS)} runs completed successfully.")
    print(f"Results saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
