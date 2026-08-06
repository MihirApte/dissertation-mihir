"""
compute_cost_single_run.py
===========================
Runs exactly ONE experiment config and measures wall-clock time and peak
GPU memory for that single run. Meant to be invoked as its own subprocess
(by compute_cost_experiment.py) so every measurement starts from a
completely clean GPU / Python process -- no leftover allocations or
cached state from a previous run can bleed into the next one's numbers.

Does NOT modify scripts/run_experiment.py in any way -- it imports and
reuses its existing, already-validated run() function exactly as-is, and
only adds timing + memory measurement around that call. This keeps the
86 already-completed dissertation experiments completely untouched.

USAGE:
    python3 compute_cost_single_run.py configs/waterglass_baseline_random.yaml

Prints exactly one line to stdout on success:
    RESULT video=<video> method=<method_suffix> wall_time_sec=<float> peak_gpu_mb=<float> frames=<int>
"""

import argparse
import os
import sys
import time

import yaml
import torch

sys.path.append(os.getcwd())
from scripts.run_experiment import run  # noqa: E402  (reused, unmodified)


def main():
    if len(sys.argv) != 2:
        print("USAGE: python3 compute_cost_single_run.py <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    input_dict = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    input_ns = argparse.Namespace(**input_dict)

    video = input_ns.video_name
    method = os.path.basename(config_path).replace(f"{video}_", "").replace(".yaml", "")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    run(input_ns)  # unmodified call into scripts/run_experiment.py
    t1 = time.time()

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else float("nan")
    )
    wall_time = t1 - t0

    # run() mutates input_ns.sample_size in place to the real frame count.
    print(
        f"RESULT video={video} method={method} wall_time_sec={wall_time:.3f} "
        f"peak_gpu_mb={peak_mb:.1f} frames={input_ns.sample_size}"
    )


if __name__ == "__main__":
    main()
