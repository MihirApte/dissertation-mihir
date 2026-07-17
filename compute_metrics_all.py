"""
compute_metrics_all.py
======================
Compares all methods across all videos for the dissertation.

Searches the results/ directory for GIFs from each method and computes
Warp Error (temporal consistency) and CLIP Score (text alignment).

USAGE:
    python3 compute_metrics_all.py
    python3 compute_metrics_all.py --device cuda   # faster CLIP on GPU
    python3 compute_metrics_all.py --skip_clip     # Warp Error only
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Method definitions — folder keyword -> label
# ---------------------------------------------------------------------------
METHODS = [
    ("baseline_random",       "Baseline (random, depth_zoe)"),
    ("semantic_shuffle",      "Semantic v1 (greedy NN, depth_zoe)"),
    ("kmeans_shuffle",        "Semantic v2 (K-means, depth_zoe)"),
    ("multicontrol_random",   "Multi-ControlNet (random, depth+canny+FreeU)"),
    ("semantic_multicontrol", "Hybrid: Semantic + Multi-ControlNet (depth+canny+FreeU)"),
]

VIDEO_PROMPTS = {
    "truck":    "Wooden trucks drive on a racetrack",
    "shanghai": "A cinematic drone view of a futuristic cyberpunk city at night",
    "street":   "People walking on a vibrant neon-lit street, anime style",
    "dog":      "An oil painting of a dog running through a sunlit green meadow",

    "airplane":      "A vintage sepia-toned photograph of an airplane taking off",
    "baseball":      "A vibrant comic book illustration of a baseball game in action",
    "bicycle":       "A watercolor painting of a bicycle riding through a city street",
    "birds":         "A soft pastel illustration of birds flying across the sky",
    "cat":           "A children's storybook illustration of a cat walking indoors",
    "cooking":       "A warm oil painting of someone cooking in a kitchen",
    "dancer":        "A dynamic charcoal sketch of a dancer performing",
    "farmland":      "A golden-hour impressionist painting of farmland",
    "fish":          "A vivid anime-style illustration of fish swimming underwater",
    "flowers":       "A delicate watercolor painting of blooming flowers",
    "forest":        "A moody, atmospheric fantasy illustration of a forest",
    "highway":       "A neon-lit synthwave illustration of a highway at dusk",
    "interview":     "A black-and-white film noir style portrait of a person talking",
    "motorboat":     "A bright pop-art illustration of a motorboat on the ocean",
    "racecars":      "A dynamic cyberpunk illustration of race cars on a track",
    "skateboarding": "A graffiti street-art style illustration of skateboarding",
    "waterglass":    "A hyperrealistic macro painting of water in a glass",
}


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def load_gif_frames(gif_path):
    from PIL import Image
    frames = []
    gif = Image.open(gif_path)
    try:
        while True:
            frames.append(np.array(gif.convert("RGB")))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames


# ---------------------------------------------------------------------------
# Warp Error
# ---------------------------------------------------------------------------
def compute_warp_error(frames):
    if len(frames) < 2:
        return float('nan')
    errors = []
    for i in range(len(frames) - 1):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        H, W = g0.shape
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        mx = (gx + flow[..., 0]).astype(np.float32)
        my = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(frames[i].astype(np.float32), mx, my,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = ((mx >= 0) & (mx < W) & (my >= 0) & (my < H)).astype(np.float32)
        diff = np.abs(warped - frames[i+1].astype(np.float32)) * mask[:,:,None]
        n = mask.sum() * 3
        errors.append(diff.sum() / n / 255.0 if n > 0 else 0.0)
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------
def compute_clip_score(frames, prompt, device='cpu'):
    import clip, torch
    from PIL import Image
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    with torch.no_grad():
        text_emb = model.encode_text(clip.tokenize([prompt]).to(device))
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    scores = []
    with torch.no_grad():
        for f in frames:
            img = preprocess(Image.fromarray(f)).unsqueeze(0).to(device)
            img_emb = model.encode_image(img)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            scores.append((img_emb * text_emb).sum().item())
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# GIF finder
# ---------------------------------------------------------------------------
def find_gif(results_dir, video, method_keyword):
    """Walk results_dir looking for a GIF whose ancestor folder contains
    both the video name and the method keyword."""
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if not f.endswith('.gif'):
                continue
            path = os.path.join(root, f)
            # Check that both video and method appear somewhere in the path
            if video in path and method_keyword in path:
                return path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--skip_clip', action='store_true')
    args = parser.parse_args()

    results = {}  # (video, method_label) -> {warp, clip}

    videos = list(VIDEO_PROMPTS.keys())

    print("\nScanning for GIFs in:", args.results_dir)
    print()

    for video in videos:
        prompt = VIDEO_PROMPTS[video]
        for keyword, label in METHODS:
            gif_path = find_gif(args.results_dir, video, keyword)
            if gif_path is None:
                print(f"  [SKIP] {video} / {label} — GIF not found")
                continue

            print(f"  [FOUND] {video} / {label}")
            print(f"          {gif_path}")
            frames = load_gif_frames(gif_path)
            print(f"          {len(frames)} frames loaded")

            warp = compute_warp_error(frames)
            clip_score = (compute_clip_score(frames, prompt, args.device)
                          if not args.skip_clip else float('nan'))

            results[(video, label)] = {'warp': warp, 'clip': clip_score}
            print(f"          Warp={warp:.4f}  CLIP={clip_score:.4f}")
            print()

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("  DISSERTATION RESULTS — All Methods x All Videos")
    print("=" * 90)
    header = f"  {'Video':<10} {'Method':<44} {'Warp':>8} {'CLIP':>8}"
    print(header)
    print("-" * 90)

    for video in videos:
        for keyword, label in METHODS:
            key = (video, label)
            if key not in results:
                print(f"  {video:<10} {label:<44} {'N/A':>8} {'N/A':>8}")
            else:
                w = results[key]['warp']
                c = results[key]['clip']
                print(f"  {video:<10} {label:<44} {w:>8.4f} {c:>8.4f}")
        print()

    print("=" * 90)
    print("  Warp Error : LOWER is better  (temporal consistency)")
    print("  CLIP Score : HIGHER is better (text-image alignment)")
    print("=" * 90)
    print()

    # Save to file
    out_path = os.path.join(args.results_dir, 'metrics_all_methods.txt')
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("RAVE Dissertation — All Methods Comparison\n\n")
        f.write(f"{'Video':<10} {'Method':<44} {'Warp':>8} {'CLIP':>8}\n")
        f.write("-" * 74 + "\n")
        for video in videos:
            for keyword, label in METHODS:
                key = (video, label)
                if key not in results:
                    f.write(f"{video:<10} {label:<44} {'N/A':>8} {'N/A':>8}\n")
                else:
                    w = results[key]['warp']
                    c = results[key]['clip']
                    f.write(f"{video:<10} {label:<44} {w:>8.4f} {c:>8.4f}\n")
            f.write("\n")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
