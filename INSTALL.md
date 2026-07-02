# Installation Guide (Blackwell GPU - RTX Pro 6000 / CUDA 12.4+)

## Step 1 - Create conda environment

```bash
conda create -n rave python=3.8
conda activate rave
conda install pip
pip cache purge
```

## Step 2 - Install PyTorch (Blackwell-compatible)

> WARNING: Do NOT use the original README's torch install command.
> The original uses CUDA 11.8 which does not support Blackwell GPUs.
> Use the command below instead.

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

## Step 3 - Install xformers (Blackwell-compatible)

```bash
pip install xformers==0.0.27
```

## Step 4 - Install all other dependencies

```bash
pip install -r requirements.txt
```

## Step 5 - Install CLIP (required for semantic shuffle improvement)

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Step 6 - Verify everything is working

```bash
python check_gpu.py
```

All checks should show [OK]. If any fail, re-read the error message -
it will tell you exactly which package to reinstall.

## Step 7 - Run experiments

```bash
bash run_experiments.sh
```

---

## Notes

- `timm==0.6.7` is pinned - do not upgrade, newer versions break the annotators.
- `diffusers==0.18.2` is pinned - the pipeline API changed in later versions.
- If `basicsr` fails to install, try: `pip install basicsr --no-build-isolation`
- If `mmdet` or `mmpose` fail, they are only used by some annotators (openpose, keypose).
  The main pipeline (depth_zoe, canny, lineart) does not need them.
