# RAVE - Semantic Shuffle Dissertation

Hi Professor, thank you for running this.

## 3 Commands to Run

```bash
# 1. Clone the repo
git clone https://github.com/mihirapte24/dissertation-mihir.git
cd dissertation-mihir

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run both experiments + metrics
bash run_experiments.sh
```

That is everything. The script runs the GPU check, baseline experiment, semantic experiment, and metrics automatically. It prints progress at each step.

---

## What to Send Back

After it finishes (roughly 30-50 min total), please send:

- `metrics_comparison.txt`
- `generated/experiment_baseline_random/*.gif`
- `generated/experiment_semantic_shuffle/*.gif`

---

## If Something Goes Wrong

| Error | Fix |
|-------|-----|
| CUDA not available | Check nvidia-smi, make sure CUDA 12.4 is active |
| xformers error | Run: pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu124 |
| clip not found | Run: pip install git+https://github.com/openai/CLIP.git |
| Out of memory | Reduce batch_size to 1 in configs/baseline_random.yaml and configs/semantic_shuffle.yaml |

---

*Mihir Apte - MSc Data Science Dissertation 2025*
