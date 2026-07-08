#!/usr/bin/env bash
# ==============================================================
#  RAVE Dissertation - Full Experiment Runner
#  Mihir Apte | MSc Data Science
# ==============================================================
#
#  Runs all experiments back to back:
#    1. Baseline (random shuffle, depth_zoe)
#    2. Semantic shuffle (CLIP greedy NN, depth_zoe)
#    3. Multi-ControlNet (random shuffle, depth_zoe + canny + FreeU)
#    4. Multi-ControlNet + Semantic (CLIP greedy NN, depth_zoe + canny + FreeU)
#
#  USAGE (from inside the RAVE folder):
#    bash run_experiments.sh [VIDEO]
#
#    VIDEO defaults to "truck". Pass another to run a different video:
#    bash run_experiments.sh shanghai
#    bash run_experiments.sh street
#    bash run_experiments.sh dog
#
#  Note: DDIM inversions are cached after first run per video.
# ==============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING: $1${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"; exit 1; }

# -- Video selection -------------------------------------------
VIDEO="${1:-truck}"
log "Running experiments for video: ${VIDEO}"
echo ""

# Validate video
if [ ! -f "data/mp4_videos/${VIDEO}.mp4" ]; then
    fail "Video not found: data/mp4_videos/${VIDEO}.mp4
    Please put your video there and re-run."
fi

# Validate configs
for cfg in "configs/${VIDEO}_baseline.yaml" "configs/${VIDEO}_semantic.yaml" \
           "configs/${VIDEO}_multicontrol.yaml"; do
    if [ ! -f "$cfg" ]; then
        warn "Config not found: $cfg — skipping that experiment."
    fi
done

# Must be run from the RAVE project root
if [ ! -f "scripts/run_experiment.py" ]; then
    fail "Run this script from inside the RAVE folder."
fi

# -- GPU check -------------------------------------------------
log "Step 0 - Verifying GPU and dependencies..."
python3 check_gpu.py || warn "GPU check reported issues. Continuing..."
echo ""

# -- Experiment 1: Baseline ------------------------------------
log "Step 1 - BASELINE (random shuffle, depth_zoe)..."
START=$SECONDS
python3 scripts/run_experiment.py configs/${VIDEO}_baseline.yaml
log "Baseline done in $(( SECONDS - START ))s  (~$(( (SECONDS - START) / 60 )) min)"
echo ""

# -- Experiment 2: Semantic Shuffle ----------------------------
log "Step 2 - SEMANTIC SHUFFLE (CLIP greedy NN, depth_zoe)..."
log "DDIM inversions cached from Step 1 — this runs faster."
START=$SECONDS
python3 scripts/run_experiment.py configs/${VIDEO}_semantic.yaml
log "Semantic done in $(( SECONDS - START ))s  (~$(( (SECONDS - START) / 60 )) min)"
echo ""

# -- Experiment 3: Multi-ControlNet ----------------------------
log "Step 3 - MULTI-CONTROLNET (random shuffle, depth_zoe + canny + FreeU)..."
log "Note: New DDIM inversion needed (different ControlNet = different cache)."
START=$SECONDS
python3 scripts/run_experiment.py configs/${VIDEO}_multicontrol.yaml
log "Multi-ControlNet done in $(( SECONDS - START ))s  (~$(( (SECONDS - START) / 60 )) min)"
echo ""

# -- Summary ---------------------------------------------------
log "=========================================================="
log "All 3 experiments complete!"
log ""
log "GIFs saved under:  results/"
log "Next step: run compute_metrics_all.py to compare all methods."
log "=========================================================="
