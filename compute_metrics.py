
import os
import sys
import argparse
import glob
import numpy as np
import cv2


def load_gif_frames(gif_path):
    """
    Load all frames from a GIF file as a list of numpy arrays (H, W, 3) uint8.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required. Run: pip install Pillow")

    frames = []
    gif = Image.open(gif_path)
    try:
        while True:
            frame = gif.convert("RGB")
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    print(f"  Loaded {len(frames)} frames from: {os.path.basename(gif_path)}")
    return frames




def compute_warp_error(frames):
    """
    Compute average warp error across all consecutive frame pairs.

    For each pair (frame_t, frame_{t+1}):
      1. Compute optical flow: frame_t -> frame_{t+1}
      2. Warp frame_t using the flow
      3. Compute masked L1 error vs frame_{t+1}

    Args:
        frames : list of np.ndarray (H, W, 3) uint8

    Returns:
        float : average warp error (lower = more temporally consistent)
    """
    if len(frames) < 2:
        print("  WARNING: Need at least 2 frames to compute warp error.")
        return float('nan')

    errors = []

    for i in range(len(frames) - 1):
        frame_t  = frames[i]
        frame_t1 = frames[i + 1]

        # Convert to grayscale for optical flow
        gray_t  = cv2.cvtColor(frame_t,  cv2.COLOR_RGB2GRAY)
        gray_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_RGB2GRAY)

        # Farneback dense optical flow - CPU, no extra install needed
        flow = cv2.calcOpticalFlowFarneback(
            gray_t, gray_t1,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        # Build pixel coordinate grid
        H, W = gray_t.shape
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

        # Compute warped coordinates
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        # Warp frame_t using the flow
        warped = cv2.remap(
            frame_t.astype(np.float32),
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Mask: exclude pixels that mapped outside the frame boundary
        valid_x = (map_x >= 0) & (map_x < W)
        valid_y = (map_y >= 0) & (map_y < H)
        mask = (valid_x & valid_y).astype(np.float32)

        # L1 error on valid pixels only
        diff = np.abs(warped - frame_t1.astype(np.float32))
        masked_diff = diff * mask[:, :, np.newaxis]
        n_valid = mask.sum() * 3  # 3 channels
        if n_valid > 0:
            error = masked_diff.sum() / n_valid / 255.0  # normalise to [0,1]
        else:
            error = 0.0

        errors.append(error)

    return float(np.mean(errors))




def compute_clip_score(frames, prompt, device='cpu'):
    """
    Compute average CLIP cosine similarity between frames and the text prompt.

    Args:
        frames : list of np.ndarray (H, W, 3) uint8
        prompt : str, the positive text prompt used for editing
        device : 'cpu' or 'cuda'

    Returns:
        float : average CLIP score in [0, 1] (higher = better alignment)
    """
    try:
        import clip
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "CLIP or torch not installed.\n"
            "Run: pip install git+https://github.com/openai/CLIP.git"
        )

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Encode text prompt once
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    scores = []
    with torch.no_grad():
        for frame_np in frames:
            pil_frame = Image.fromarray(frame_np)
            img_tensor = preprocess(pil_frame).unsqueeze(0).to(device)
            img_embedding = model.encode_image(img_tensor)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)

            # Cosine similarity (dot product since both are L2-normalised)
            score = (img_embedding * text_embedding).sum().item()
            scores.append(score)

    return float(np.mean(scores))




def find_gif(results_dir, keyword):
    """
    Search results_dir recursively for a GIF whose parent folder name
    contains keyword. Returns the first match.
    """
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith('.gif') and keyword in root:
                return os.path.join(root, f)
    return None




def print_results_table(baseline_warp, semantic_warp,
                        baseline_clip, semantic_clip):

    warp_improvement = ((baseline_warp - semantic_warp) / baseline_warp * 100
                        if baseline_warp > 0 else 0)
    clip_improvement = ((semantic_clip - baseline_clip) / baseline_clip * 100
                        if baseline_clip > 0 else 0)

    warp_winner = "Semantic [OK]" if semantic_warp < baseline_warp else "Baseline"
    clip_winner = "Semantic [OK]" if semantic_clip > baseline_clip else "Baseline"

    print("\n" + "=" * 62)
    print("  RAVE Dissertation - Metric Comparison")
    print("=" * 62)
    print(f"  {'Metric':<28} {'Baseline':>10} {'Semantic':>10} {'Winner':>10}")
    print("-" * 62)
    print(f"  {'Warp Error (lower=better)':<28} {baseline_warp:>10.4f} {semantic_warp:>10.4f} {warp_winner:>10}")
    print(f"  {'CLIP Score (higher=better)':<28} {baseline_clip:>10.4f} {semantic_clip:>10.4f} {clip_winner:>10}")
    print("-" * 62)
    print(f"  Warp Error improvement : {warp_improvement:+.2f}%  (negative = semantic better)")
    print(f"  CLIP Score improvement : {clip_improvement:+.2f}%  (positive = semantic better)")
    print("=" * 62)
    print()




def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute warp error and CLIP score for RAVE experiments."
    )
    parser.add_argument('--baseline',     type=str, default=None,
                        help='Path to baseline (random shuffle) GIF')
    parser.add_argument('--semantic',     type=str, default=None,
                        help='Path to semantic shuffle GIF')
    parser.add_argument('--results_dir',  type=str, default='results',
                        help='Results folder - used for auto-discovery if '
                             '--baseline / --semantic not provided')
    parser.add_argument('--prompt',       type=str,
                        default='Wooden trucks drive on a racetrack',
                        help='Text prompt used during editing (for CLIP score)')
    parser.add_argument('--device',       type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for CLIP inference (default: cpu)')
    parser.add_argument('--skip_clip',    action='store_true',
                        help='Skip CLIP score (faster, no CLIP model needed)')
    return parser.parse_args()


def main():
    args = parse_args()

    # -- Resolve GIF paths -------------------------------------
    baseline_gif = args.baseline
    semantic_gif = args.semantic

    if baseline_gif is None:
        print(f"[Auto-discover] Searching '{args.results_dir}' for baseline GIF...")
        baseline_gif = find_gif(args.results_dir, 'baseline_random')
        if baseline_gif is None:
            baseline_gif = find_gif(args.results_dir, 'random')
        if baseline_gif is None:
            print("ERROR: Could not find baseline GIF. "
                  "Pass --baseline <path> explicitly.")
            sys.exit(1)
        print(f"  Found: {baseline_gif}")

    if semantic_gif is None:
        print(f"[Auto-discover] Searching '{args.results_dir}' for semantic GIF...")
        semantic_gif = find_gif(args.results_dir, 'semantic')
        if semantic_gif is None:
            print("ERROR: Could not find semantic GIF. "
                  "Pass --semantic <path> explicitly.")
            sys.exit(1)
        print(f"  Found: {semantic_gif}")

    # -- Load frames -------------------------------------------
    print("\n[1] Loading frames...")
    baseline_frames = load_gif_frames(baseline_gif)
    semantic_frames = load_gif_frames(semantic_gif)

    # -- Warp Error --------------------------------------------
    print("\n[2] Computing Warp Error (temporal consistency)...")
    print("  Baseline  :", end=' ', flush=True)
    baseline_warp = compute_warp_error(baseline_frames)
    print(f"{baseline_warp:.4f}")

    print("  Semantic  :", end=' ', flush=True)
    semantic_warp = compute_warp_error(semantic_frames)
    print(f"{semantic_warp:.4f}")

    # -- CLIP Score --------------------------------------------
    if args.skip_clip:
        print("\n[3] CLIP Score skipped (--skip_clip flag set).")
        baseline_clip = float('nan')
        semantic_clip = float('nan')
    else:
        print(f"\n[3] Computing CLIP Score (prompt: '{args.prompt}')...")
        print(f"    Device: {args.device}")
        print("  Baseline  :", end=' ', flush=True)
        baseline_clip = compute_clip_score(baseline_frames, args.prompt, args.device)
        print(f"{baseline_clip:.4f}")

        print("  Semantic  :", end=' ', flush=True)
        semantic_clip = compute_clip_score(semantic_frames, args.prompt, args.device)
        print(f"{semantic_clip:.4f}")

    # -- Results Table -----------------------------------------
    print_results_table(baseline_warp, semantic_warp,
                        baseline_clip, semantic_clip)

    # -- Save to file ------------------------------------------
    out_path = os.path.join(args.results_dir, 'metrics_comparison.txt')
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"Baseline GIF : {baseline_gif}\n")
        f.write(f"Semantic GIF : {semantic_gif}\n")
        f.write(f"Prompt       : {args.prompt}\n\n")
        f.write(f"Warp Error   - Baseline: {baseline_warp:.4f}  |  Semantic: {semantic_warp:.4f}\n")
        f.write(f"CLIP Score   - Baseline: {baseline_clip:.4f}  |  Semantic: {semantic_clip:.4f}\n")

    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
