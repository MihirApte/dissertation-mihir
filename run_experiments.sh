#!/usr/bin/env bash
# ==============================================================
#  RAVE Dissertation - Full Experiment Runner
#  Mihir Apte | MSc Data Science
# ==============================================================
#
#  Runs both experiments (baseline + semantic shuffle) back to back,
#  then computes metrics and prints a comparison table.
#
#  USAGE (from inside the RAVE folder):
#    conda activate rave
#    bash run_experiments.sh
#
#  IMPORTANT:
#    - Run check_gpu.py first to make sure everything is set up.
#    - Put your video at:  data/mp4_videos/truck.mp4
#    - DDIM inversion results are cached after Experiment 1,
#      so Experiment 2 reuses them and runs faster.
#
# ==============================================================

set -e  # Stop immediately if any command fails

# -- Colour helpers --------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No colour

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"; exit 1; }

# -- Sanity checks ---------------------------------------------
log "Starting RAVE experiment runner..."
echo ""

# Must be run from the RAVE project root
if [ ! -f "scripts/run_experiment.py" ]; then
    fail "This script must be run from inside the RAVE project folder.
    cd into the RAVE folder first, then run:  bash run_experiments.sh"
fi

# Check video exists
VIDEO="data/mp4_videos/truck.mp4"
if [ ! -f "$VIDEO" ]; then
    fail "Input video not found at: $VIDEO
    Please put your video there and re-run."
fi

# -- Step 0: GPU check -----------------------------------------
log "Step 0 - Verifying GPU and dependencies..."
python check_gpu.py || fail "GPU check failed. Fix the issues above before continuing."
echo ""

# -- Step 1: Baseline experiment (random shuffle) --------------
log "Step 1 - Running BASELINE experiment (original random shuffle)..."
log "Config : configs/baseline_random.yaml"
echo ""

START_TIME_1=$SECONDS
python scripts/run_experiment.py configs/baseline_random.yaml
DURATION_1=$(( SECONDS - START_TIME_1 ))

echo ""
log "Baseline experiment complete. Time: ${DURATION_1}s  (~$(( DURATION_1 / 60 )) min)"
echo ""

# -- Step 2: Semantic shuffle experiment -----------------------
log "Step 2 - Running SEMANTIC SHUFFLE experiment (dissertation improvement)..."
log "Config : configs/semantic_shuffle.yaml"
log "Note   : DDIM inversions are cached from Step 1 - this run will be faster."
echo ""

START_TIME_2=$SECONDS
python scripts/run_experiment.py configs/semantic_shuffle.yaml
DURATION_2=$(( SECONDS - START_TIME_2 ))

echo ""
log "Semantic experiment complete. Time: ${DURATION_2}s  (~$(( DURATION_2 / 60 )) min)"
echo ""

# -- Step 3: Compute metrics -----------------------------------
log "Step 3 - Computing evaluation metrics (Warp Error + CLIP Score)..."
echo ""

python compute_metrics.py \
    --results_dir results \
    --prompt "Wooden trucks drive on a racetrack" \
    --device cpu

echo ""

# -- Summary ---------------------------------------------------
TOTAL=$(( SECONDS ))
echo ""
log "=========================================================="
log "All done!"
log "Total runtime : ~$(( TOTAL / 60 )) minutes"
log ""
log "Output GIFs saved under:  results/"
log "Metric comparison saved:  results/metrics_comparison.txt"
log "=========================================================="
