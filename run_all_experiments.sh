#!/usr/bin/env bash
# ==============================================================
#  RAVE Dissertation - MASTER Experiment Runner
#  Mihir Apte | MSc Data Science | TCD
# ==============================================================
#
#  Runs ALL experiments for ALL videos in one go:
#    For each video:
#      1. Baseline       (random shuffle,   depth_zoe)
#      2. Semantic v1    (greedy NN,        depth_zoe)
#      3. Semantic v2    (K-means,          depth_zoe)
#      4. Multi-Control  (random shuffle,   depth_zoe + canny + FreeU)
#
#  Then computes a full comparison table across all methods and videos.
#
#  SETUP (one time):
#    pip install -r requirements.txt
#    pip install git+https://github.com/openai/CLIP.git scikit-learn
#
#  Place videos at:
#    data/mp4_videos/truck.mp4
#    data/mp4_videos/shanghai.mp4
#    data/mp4_videos/street.mp4
#    data/mp4_videos/dog.mp4
#
#  USAGE (from inside the RAVE folder):
#    bash run_all_experiments.sh
#
#  To run only specific videos, comment out blocks below.
#
#  NOTE: DDIM inversions are cached per video after the first experiment,
#        so experiments 2 and 3 for each video run faster.
#        Multi-Control uses a separate cache (different ControlNet).
# ==============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()     { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"; }
section() { echo -e "\n${CYAN}======================================================${NC}"; \
            echo -e "${CYAN}  $1${NC}"; \
            echo -e "${CYAN}======================================================${NC}\n"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"; }
fail()    { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"; exit 1; }

run_exp() {
    local config="$1"
    local label="$2"
    if [ ! -f "$config" ]; then
        warn "Config not found: $config — skipping."
        return 0
    fi
    log "Running: $label"
    local start=$SECONDS
    python3 scripts/run_experiment.py "$config"
    log "Done in $(( SECONDS - start ))s  (~$(( (SECONDS - start) / 60 )) min)"
    echo ""
}

# -- Pre-flight checks -----------------------------------------
if [ ! -f "scripts/run_experiment.py" ]; then
    fail "Run this script from inside the RAVE folder (cd dissertation-mihir)"
fi

section "RAVE Dissertation — Full Experiment Suite"
log "GPU check..."
python3 check_gpu.py || warn "GPU check had issues — continuing anyway."
echo ""

# ==============================================================
#  TRUCK
# ==============================================================
if [ -f "data/mp4_videos/truck.mp4" ]; then
    section "VIDEO: truck"
    run_exp "configs/baseline_random.yaml"       "Truck — Baseline (random shuffle)"
    run_exp "configs/semantic_shuffle.yaml"      "Truck — Semantic v1 (greedy NN)  [cache reused]"
    run_exp "configs/truck_kmeans.yaml"          "Truck — Semantic v2 (K-means)    [cache reused]"
    run_exp "configs/truck_multicontrol.yaml"    "Truck — Multi-ControlNet + FreeU  [new cache]"
else
    warn "truck.mp4 not found — skipping truck experiments."
fi

# ==============================================================
#  SHANGHAI
# ==============================================================
if [ -f "data/mp4_videos/shanghai.mp4" ]; then
    section "VIDEO: shanghai"
    run_exp "configs/shanghai_baseline.yaml"     "Shanghai — Baseline (random shuffle)"
    run_exp "configs/shanghai_semantic.yaml"     "Shanghai — Semantic v1 (greedy NN)  [cache reused]"
    run_exp "configs/shanghai_kmeans.yaml"       "Shanghai — Semantic v2 (K-means)    [cache reused]"
    run_exp "configs/shanghai_multicontrol.yaml" "Shanghai — Multi-ControlNet + FreeU  [new cache]"
else
    warn "shanghai.mp4 not found — skipping shanghai experiments."
fi

# ==============================================================
#  STREET
# ==============================================================
if [ -f "data/mp4_videos/street.mp4" ]; then
    section "VIDEO: street"
    run_exp "configs/street_baseline.yaml"       "Street — Baseline (random shuffle)"
    run_exp "configs/street_semantic.yaml"       "Street — Semantic v1 (greedy NN)  [cache reused]"
    run_exp "configs/street_kmeans.yaml"         "Street — Semantic v2 (K-means)    [cache reused]"
    run_exp "configs/street_multicontrol.yaml"   "Street — Multi-ControlNet + FreeU  [new cache]"
else
    warn "street.mp4 not found — skipping street experiments."
fi

# ==============================================================
#  DOG
# ==============================================================
if [ -f "data/mp4_videos/dog.mp4" ]; then
    section "VIDEO: dog"
    run_exp "configs/dog_baseline.yaml"          "Dog — Baseline (random shuffle)"
    run_exp "configs/dog_semantic.yaml"          "Dog — Semantic v1 (greedy NN)  [cache reused]"
    run_exp "configs/dog_kmeans.yaml"            "Dog — Semantic v2 (K-means)    [cache reused]"
    run_exp "configs/dog_multicontrol.yaml"      "Dog — Multi-ControlNet + FreeU  [new cache]"
else
    warn "dog.mp4 not found — skipping dog experiments."
fi

# ==============================================================
#  METRICS
# ==============================================================
section "Computing Metrics — All Methods x All Videos"
python3 compute_metrics_all.py --device cuda || \
python3 compute_metrics_all.py --device cpu

# ==============================================================
#  DONE
# ==============================================================
section "All Done!"
log "GIF results saved under:      results/"
log "Full metrics table saved to:  results/metrics_all_methods.txt"
log ""
log "To download results from Colab:"
log "  import shutil"
log "  shutil.make_archive('rave_results', 'zip', 'results')"
log "  from google.colab import files; files.download('rave_results.zip')"
