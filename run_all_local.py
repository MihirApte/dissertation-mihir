"""
run_all_local.py
=================
Resumable runner for the full experiment set - discovers every config in
configs/, and runs them in a safety-first order: all baseline configs
first, then all semantic v1, then all K-means, then all Multi-ControlNet,
then the hybrid configs last. That way, if time runs out partway through,
every video already has its safest/most informative methods done rather
than some videos having all 4 and others having none.

Safe to re-run after any interruption (Ctrl+C, disconnect, crash): before
running each config, it checks whether a result GIF already exists for
that config's save_folder, and skips it if so - so re-running this after
a partial run only does the remaining work.

USAGE (run from inside the RAVE project folder, with your conda env active):
    python3 run_all_local.py
"""

import glob
import os
import subprocess
import sys
import threading
import time
import yaml

CONFIG_DIR = "configs"
VIDEO_DIR = "data/mp4_videos"

# configs/ also contains a handful of stray example files left over from
# the original RAVE repo (truck.yaml, truck-multicontrolnet.yaml, etc.)
# that were never part of this dissertation's experiment design and would
# fail or collide (they all share save_folder: 'truck'). Rather than glob
# configs/*.yaml and guess by filename suffix - which also silently
# mis-sorted dog/shanghai/street's configs, since those use a shorter
# naming convention than the new videos - we use an explicit allowlist of
# exactly the files this project actually runs.

# The 4 original videos each used a slightly different naming convention
# depending on when they were created - listed explicitly rather than
# pattern-matched.
LEGACY_CONFIGS = [
    (f"{CONFIG_DIR}/baseline_random.yaml",       "Baseline"),
    (f"{CONFIG_DIR}/semantic_shuffle.yaml",      "Semantic v1"),
    (f"{CONFIG_DIR}/truck_kmeans.yaml",          "Semantic v2 (K-means)"),
    (f"{CONFIG_DIR}/truck_multicontrol.yaml",    "Multi-ControlNet"),

    (f"{CONFIG_DIR}/dog_baseline.yaml",          "Baseline"),
    (f"{CONFIG_DIR}/dog_semantic.yaml",          "Semantic v1"),
    (f"{CONFIG_DIR}/dog_kmeans.yaml",            "Semantic v2 (K-means)"),
    (f"{CONFIG_DIR}/dog_multicontrol.yaml",      "Multi-ControlNet"),

    (f"{CONFIG_DIR}/shanghai_baseline.yaml",     "Baseline"),
    (f"{CONFIG_DIR}/shanghai_semantic.yaml",     "Semantic v1"),
    (f"{CONFIG_DIR}/shanghai_kmeans.yaml",       "Semantic v2 (K-means)"),
    (f"{CONFIG_DIR}/shanghai_multicontrol.yaml", "Multi-ControlNet"),

    (f"{CONFIG_DIR}/street_baseline.yaml",       "Baseline"),
    (f"{CONFIG_DIR}/street_semantic.yaml",       "Semantic v1"),
    (f"{CONFIG_DIR}/street_kmeans.yaml",         "Semantic v2 (K-means)"),
    (f"{CONFIG_DIR}/street_multicontrol.yaml",   "Multi-ControlNet"),
]

HYBRID_CONFIGS = [
    (f"{CONFIG_DIR}/dog_semantic_multicontrol.yaml",      "Hybrid (Semantic + Multi-ControlNet)"),
    (f"{CONFIG_DIR}/shanghai_semantic_multicontrol.yaml", "Hybrid (Semantic + Multi-ControlNet)"),
]

# The 17 videos added via prepare_sample_videos.py, all following the
# systematic {video}_{method}.yaml naming produced by generate_configs.py.
NEW_VIDEOS = [
    "airplane", "baseball", "bicycle", "birds", "cat", "cooking", "dancer",
    "farmland", "fish", "flowers", "forest", "highway", "interview",
    "motorboat", "racecars", "skateboarding", "waterglass",
]


def discover_configs():
    """Explicit allowlist, ordered method-major: all baseline configs
    first, then all semantic v1, then all K-means, then all
    Multi-ControlNet, hybrid last. 16 legacy + 68 new + 2 hybrid = 86."""
    ordered = []

    for label in ("Baseline", "Semantic v1", "Semantic v2 (K-means)", "Multi-ControlNet"):
        ordered += [(p, l) for p, l in LEGACY_CONFIGS if l == label]
        suffix = {
            "Baseline": "baseline_random",
            "Semantic v1": "semantic_shuffle",
            "Semantic v2 (K-means)": "kmeans_shuffle",
            "Multi-ControlNet": "multicontrol_random",
        }[label]
        ordered += [(f"{CONFIG_DIR}/{v}_{suffix}.yaml", label) for v in NEW_VIDEOS]

    ordered += HYBRID_CONFIGS

    missing = [p for p, _ in ordered if not os.path.exists(p)]
    if missing:
        print("WARNING - expected config file(s) not found, skipping:")
        for m in missing:
            print(f"  {m}")

    return [(p, l) for p, l in ordered if os.path.exists(p)]


def already_done(save_folder):
    # actual layout is results/{date}/{save_folder}/{video_name}/{prompt-index}/*.gif
    # - one more directory level (video_name) than this used to check for.
    return len(glob.glob(f"results/*/{save_folder}/*/*/*.gif")) > 0


def video_missing(cfg):
    video_path = os.path.join(VIDEO_DIR, f"{cfg['video_name']}.mp4")
    return not os.path.exists(video_path)


def stream_run(cmd, heartbeat_secs=30):
    """Run a subprocess, streaming its raw output live, with a heartbeat
    message if it goes quiet for a while (model downloads, DDIM inversion
    setup can be silent for minutes before any progress bar appears)."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    last_output = time.time()
    stop = threading.Event()

    def watchdog():
        while not stop.is_set():
            time.sleep(5)
            idle = time.time() - last_output
            if idle > heartbeat_secs and not stop.is_set():
                print(f"   ... still running, no new output for {int(idle)}s "
                      f"(normal during model download / DDIM inversion setup)", flush=True)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            sys.stdout.write(chunk.decode(errors="replace"))
            sys.stdout.flush()
            last_output = time.time()
    finally:
        stop.set()
        process.stdout.close()
    return process.wait()


def run_config(config_path, label, idx, total):
    tag = f"[{idx}/{total}]"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if video_missing(cfg):
        print(f"{tag} [SKIP] {config_path}: source video '{cfg['video_name']}.mp4' "
              f"missing from {VIDEO_DIR}/")
        return "skip"

    save_folder = cfg["save_folder"]
    if already_done(save_folder):
        print(f"{tag} [SKIP] {config_path} ({label}): already completed")
        return "done"

    print(f"\n{'='*70}\n{tag} [RUN] {config_path}  ({label})\n{'='*70}", flush=True)
    t0 = time.time()
    # use the exact same interpreter running this script (sys.executable),
    # not a hardcoded "python3" - on Windows/conda that name can resolve to
    # a different Python install with no packages installed at all.
    ret = stream_run([sys.executable, "scripts/run_experiment.py", config_path])
    mins = (time.time() - t0) / 60
    if ret == 0:
        print(f"{tag} [OK] finished in {mins:.1f} min", flush=True)
        return "ok"
    else:
        print(f"{tag} [FAIL] exit code {ret} - continuing to next experiment", flush=True)
        return "fail"


def main():
    experiments = discover_configs()
    total = len(experiments)
    print(f"{total} config(s) queued.\n")

    summary = {}
    run_start = time.time()
    for i, (config_path, label) in enumerate(experiments, 1):
        summary[config_path] = run_config(config_path, label, i, total)
        done_count = sum(1 for s in summary.values() if s in ("ok", "done"))
        elapsed = (time.time() - run_start) / 60
        print(f"--- progress: {done_count}/{total} accounted for | {elapsed:.1f} min elapsed ---\n", flush=True)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for path, status in summary.items():
        print(f"  [{status.upper():<5}] {path}")

    failed = [p for p, s in summary.items() if s == "fail"]
    if failed:
        print(f"\n{len(failed)} experiment(s) failed - re-run this script to retry them "
              f"(completed ones will be skipped).")


if __name__ == "__main__":
    main()
