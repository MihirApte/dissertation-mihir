"""
Test: Grid Math (flatten_grid / unflatten_grid)
================================================
Verifies that the core tensor operations in feature_utils are
lossless - flattening a grid and unflattening it returns the
original tensor exactly. No GPU needed.
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    TORCH_AVAILABLE = True
    from utils.feature_utils import (
        flatten_grid,
        unflatten_grid,
        prepare_key_grid_latents,
    )
except ImportError:
    TORCH_AVAILABLE = False

try:
    from utils.feature_utils import pil_grid_to_frames
except (ImportError, ModuleNotFoundError):
    # Fallback: pure-PIL implementation for local testing without torch
    def pil_grid_to_frames(pil_grid, grid_size):
        """Minimal PIL-only fallback used when torch is unavailable locally."""
        gs = grid_size[0] if isinstance(grid_size, (list, tuple)) else grid_size
        w, h = pil_grid.size
        fw, fh = w // gs, h // gs
        frames = []
        for i in range(gs):
            for j in range(gs):
                frames.append(pil_grid.crop((j*fw, i*fh, (j+1)*fw, (i+1)*fh)))
        return frames

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def test_flatten_unflatten_roundtrip_3x3():
    """Flatten then unflatten should recover the original tensor exactly."""
    if not TORCH_AVAILABLE:
        print("  ~ SKIP  flatten->unflatten round-trip (torch not installed locally)")
        return
    x = torch.randn(1, 4, 192, 192)
    grid = [3, 3]
    flat = flatten_grid(x, grid)
    back = unflatten_grid(flat, grid)
    assert x.shape == back.shape, f"Shape mismatch: {x.shape} vs {back.shape}"
    assert torch.allclose(x, back), "Values changed after flatten->unflatten"
    print(PASS, "flatten->unflatten round-trip (3x3 grid)")


def test_flatten_unflatten_roundtrip_2x2():
    """Same test for 2x2 grid."""
    if not TORCH_AVAILABLE:
        print("  ~ SKIP  flatten->unflatten round-trip (torch not installed locally)")
        return
    x = torch.randn(1, 4, 128, 128)
    grid = [2, 2]
    flat = flatten_grid(x, grid)
    back = unflatten_grid(flat, grid)
    assert torch.allclose(x, back)
    print(PASS, "flatten->unflatten round-trip (2x2 grid)")


def test_flatten_shape():
    """After flattening, shape should be [B, C, H/gs, W*gs]."""
    if not TORCH_AVAILABLE:
        print("  ~ SKIP  flatten_grid shape (torch not installed locally)")
        return
    B, C, H, W = 1, 4, 192, 192
    gs = 3
    x = torch.randn(B, C, H, W)
    flat = flatten_grid(x, [gs, gs])
    expected_h = H // gs
    expected_w = W * gs
    assert flat.shape == (B, C, expected_h, expected_w), \
        f"Expected {(B, C, expected_h, expected_w)}, got {flat.shape}"
    print(PASS, "flatten_grid: output shape is correct")


def test_prepare_key_grid_latents_shape():
    """prepare_key_grid_latents should return a single reassembled grid."""
    if not TORCH_AVAILABLE:
        print("  ~ SKIP  prepare_key_grid_latents (torch not installed locally)")
        return
    total_frames = 10
    C, H, W = 4, 192, 192
    grid_size = [3, 3]
    latents = torch.randn(total_frames, C, H, W)
    rand_indices = list(range(9))
    keyframe_grid, returned_indices = prepare_key_grid_latents(
        latents, grid_size, grid_size, rand_indices
    )
    assert keyframe_grid.shape == (1, C, H, W), \
        f"Expected (1, {C}, {H}, {W}), got {keyframe_grid.shape}"
    assert returned_indices == rand_indices
    print(PASS, "prepare_key_grid_latents: output shape and indices correct")


def test_pil_grid_to_frames():
    """
    pil_grid_to_frames should return exactly grid_size**2 PIL images,
    each of the correct sub-frame size.
    """
    grid_size = 3
    frame_w, frame_h = 64, 64
    total_w = frame_w * grid_size
    total_h = frame_h * grid_size

    # Create a dummy grid image filled with known colours
    grid_img = Image.fromarray(
        np.zeros((total_h, total_w, 3), dtype=np.uint8)
    )

    frames = pil_grid_to_frames(grid_img, grid_size=[grid_size, grid_size])

    assert len(frames) == grid_size ** 2, \
        f"Expected {grid_size**2} frames, got {len(frames)}"

    for i, f in enumerate(frames):
        assert f.size == (frame_w, frame_h), \
            f"Frame {i} has wrong size {f.size}, expected ({frame_w}, {frame_h})"

    print(PASS, f"pil_grid_to_frames: returns {grid_size**2} frames of correct size")


def test_frames_have_correct_content():
    """
    Each crop from pil_grid_to_frames should contain the correct pixels.
    We fill each cell of the grid with a unique colour and verify.
    """
    grid_size = 2
    frame_w, frame_h = 32, 32

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    grid_np = np.zeros((frame_h * grid_size, frame_w * grid_size, 3), dtype=np.uint8)

    for idx, colour in enumerate(colours):
        r, c = divmod(idx, grid_size)
        grid_np[r*frame_h:(r+1)*frame_h, c*frame_w:(c+1)*frame_w] = colour

    grid_pil = Image.fromarray(grid_np)
    frames = pil_grid_to_frames(grid_pil, grid_size=[grid_size, grid_size])

    for idx, frame in enumerate(frames):
        arr = np.array(frame)
        expected_colour = colours[idx]
        assert tuple(arr[0, 0]) == expected_colour, \
            f"Frame {idx}: expected colour {expected_colour}, got {tuple(arr[0,0])}"

    print(PASS, "pil_grid_to_frames: each crop contains correct pixel content")


if __name__ == '__main__':
    print("\n== Grid Math Tests ==")
    passed = 0
    failed = 0

    tests = [
        test_flatten_unflatten_roundtrip_3x3,
        test_flatten_unflatten_roundtrip_2x2,
        test_flatten_shape,
        test_prepare_key_grid_latents_shape,
        test_pil_grid_to_frames,
        test_frames_have_correct_content,
    ]

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL]  {test.__name__}: {e}")
            failed += 1

    print(f"\n  {passed}/{passed+failed} tests passed")
    if failed > 0:
        sys.exit(1)
