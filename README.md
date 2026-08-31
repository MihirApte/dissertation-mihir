# Improving Zero-Shot Video Editing with Diffusion Models

MSc Data Science dissertation project by **Mihir Apte**, Trinity College Dublin, School of Computer Science and Statistics (September 2026).

This project builds on **RAVE** (Randomized noise shuffling for fast and consistent video editing, CVPR 2024), a training-free, zero-shot text-guided video editing framework. It proposes and evaluates three training-free improvements to RAVE's pipeline, plus a hybrid combination, across 21 test videos and 86 experiments.

## What this project adds on top of RAVE

- **Semantic Shuffle v1** - replaces RAVE's random frame grouping with a CLIP-embedding-guided greedy nearest-neighbour grouping, so frames sharing a grid are genuinely similar rather than random.
- **Semantic Shuffle v2** - the same idea, but grouping is done by K-means clustering over the whole video's CLIP embeddings at once, instead of a local, step-by-step decision.
- **Multi-ControlNet + FreeU** - adds a second ControlNet condition (Canny edges, alongside depth) and FreeU feature reweighting, for richer structural guidance.
- **Hybrid experiment** - combines semantic shuffling with Multi-ControlNet, tested and reported honestly (it does not outperform Multi-ControlNet alone).

## Evaluation

All four core methods were compared on 21 videos (86 experiments total) using two automatic metrics - **Warp Error** (temporal consistency, via Farnebäck optical flow) and **CLIP Score** (text-prompt alignment) - plus a **computational-cost comparison** (wall-clock time and peak GPU memory on 5 videos) and a **blind third-party perceptual survey** (5 videos, 5 reviewers).

Headline findings:
- No single method wins everywhere. Baseline is still the hardest method to beat on raw motion smoothness for a good number of videos.
- Semantic Shuffle (v1 and v2) gives smaller but consistent gains, at essentially no extra computational cost.
- Multi-ControlNet + FreeU gives the largest gains and is backed by a majority of blind survey reviewers, at roughly 24% more runtime and 11% more GPU memory than Baseline.
- The hybrid combination does not outperform Multi-ControlNet alone.

Full methodology, all 21-video results, and the complete discussion are in the dissertation report (not included in this repository).

## Repository structure

```
configs/         Per-video, per-method experiment configs (yaml)
scripts/         Core experiment-running code (built on RAVE's pipeline)
pipelines/       Diffusion pipeline implementations
annotator/       Structural conditioning extractors (depth, Canny, etc.) - from the original RAVE repo
utils/           Shared utility code
compute_cost_experiment.py    Orchestrates the computational-cost comparison (5 videos x 4 methods)
compute_cost_single_run.py    Runs one config as an isolated subprocess, measuring wall time + peak GPU memory
compute_metrics_all.py        Computes Warp Error and CLIP Score across all experiments
generate_results_deepdive_figures.py   Generates the report's win-count, heatmap, cost, and survey figures
check_gpu.py      Verifies the environment/GPU setup before running experiments
run_experiments.sh   Runs the baseline / semantic / multi-controlnet experiments for one video
INSTALL.md        Environment setup instructions
```

## Setup

See `INSTALL.md` for full environment setup (Python 3.8, PyTorch, `diffusers==0.18.2`, `xformers`, CLIP). All experiments reported in the dissertation were run on a single NVIDIA A6000 GPU (49GB VRAM).

```bash
pip install -r requirements.txt
python check_gpu.py
bash run_experiments.sh <video_name>
```

## Acknowledgement

This project builds directly on the official RAVE implementation:

> Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M. Rehg, Pinar Yanardag. **RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models.** CVPR 2024.
> [arXiv:2312.04524](https://arxiv.org/abs/2312.04524) · [Project page](https://rave-video.github.io/) · [Original repository](https://github.com/rehg-lab/RAVE)

All credit for the base RAVE framework (the grid trick, noise shuffling, and the underlying pipeline code this project extends) belongs to the original authors. The `annotator/` directory and core pipeline scaffolding in this repository are from their implementation.
