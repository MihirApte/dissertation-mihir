
import random
import numpy as np
from PIL import Image



def load_clip_model(device='cpu'):
    """
    Load CLIP ViT-B/32. Downloads weights (~350MB) on first call,
    cached locally afterwards.
    """
    import torch  # noqa: imported here to allow numpy-only local testing
    try:
        import clip
    except ImportError:
        raise ImportError(
            "openai-clip is not installed.\n"
            "Run: pip install git+https://github.com/openai/CLIP.git"
        )
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def compute_frame_embeddings(image_pil_list, grid_size, device='cpu'):
    """
    Compute L2-normalised CLIP embeddings for every individual frame
    across all grid images. Called once before the denoising loop.

    Args:
        image_pil_list : list of PIL images, each a (grid_size x grid_size)
                         tiled grid of video frames.
        grid_size      : int, e.g. 3 for a 3x3 grid.
        device         : 'cpu' (default) or 'cuda'. CPU is fine - this
                         runs once and CLIP is small.

    Returns:
        embeddings : torch.Tensor of shape [total_frames, 512],
                     L2-normalised, on CPU.
    """
    import torch  
    model, preprocess = load_clip_model(device=device)

    all_embeddings = []

    with torch.no_grad():
        for grid_pil in image_pil_list:
            # Split the grid image into individual frame PIL images
            frames = _pil_grid_to_frames(grid_pil, grid_size)
            for frame_pil in frames:
                # CLIP expects RGB PIL -> preprocess -> [1, 3, 224, 224]
                img_tensor = preprocess(frame_pil).unsqueeze(0).to(device)
                embedding = model.encode_image(img_tensor)
                # L2 normalise so cosine similarity = dot product
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                all_embeddings.append(embedding.squeeze(0).cpu().float())

    embeddings = torch.stack(all_embeddings)  # [N, 512]
    return embeddings


def _pil_grid_to_frames(pil_grid, grid_size):
    """
    Split a PIL grid image into individual frame PIL images.
    Mirrors feature_utils.pil_grid_to_frames() to avoid circular imports.
    """
    w, h = pil_grid.size
    img_w = w // grid_size
    img_h = h // grid_size
    frames = []
    for i in range(grid_size):
        for j in range(grid_size):
            crop = pil_grid.crop((j * img_w, i * img_h,
                                   (j + 1) * img_w, (i + 1) * img_h))
            frames.append(crop)
    return frames




def cosine_similarity_matrix(embeddings):
    """
    Compute pairwise cosine similarity matrix from L2-normalised embeddings.
    Accepts either a torch.Tensor or a numpy.ndarray.

    Args:
        embeddings : torch.Tensor or np.ndarray [N, D], L2-normalised.

    Returns:
        sim_matrix : np.ndarray [N, N], values in [-1, 1].
    """
    # Normalise if not already (safe to call twice)
    if hasattr(embeddings, 'numpy'):
        # torch.Tensor path
        arr = embeddings.numpy().astype(np.float32)
    else:
        arr = np.array(embeddings, dtype=np.float32)

    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    arr = arr / norms

    sim = arr @ arr.T
    return sim




def semantic_permutation(embeddings, grid_frame_number, total_frame_number):
    """
    Generate a permutation of frame indices where visually similar frames
    are assigned to the same grid slot.

    At each denoising step this is called fresh (with the random module's
    current state), so different steps produce different groupings while
    always respecting visual similarity.

    Args:
        embeddings        : torch.Tensor [N, D], L2-normalised CLIP embeddings.
        grid_frame_number : int, frames per grid (e.g. 9 for a 3x3 grid).
        total_frame_number: int, total number of frames N.

    Returns:
        permutation : list of N int indices - a valid permutation of
                      range(total_frame_number) with similar frames
                      grouped into consecutive blocks of grid_frame_number.
    """
    sim_matrix = cosine_similarity_matrix(embeddings)  # [N, N]

    unassigned = list(range(total_frame_number))
    permutation = []

    while len(unassigned) >= grid_frame_number:
        # 1. Pick a random seed frame from what's left
        seed_frame = random.choice(unassigned)

        # 2. Rank remaining frames by cosine similarity to the seed
        scores = [(sim_matrix[seed_frame, j], j) for j in unassigned]
        scores.sort(key=lambda x: x[0], reverse=True)

        # 3. Take the top grid_frame_number most similar frames as one group
        group = [j for _, j in scores[:grid_frame_number]]
        permutation.extend(group)

        # 4. Remove assigned frames from the pool
        assigned_set = set(group)
        unassigned = [j for j in unassigned if j not in assigned_set]

    # Append any leftover frames (edge case: total not divisible by grid size)
    permutation.extend(unassigned)

    return permutation




def validate_permutation(permutation, total_frame_number):
    """
    Assert that a permutation is valid:
    - Correct length
    - Contains every index exactly once
    """
    assert len(permutation) == total_frame_number, (
        f"Permutation length {len(permutation)} != {total_frame_number}"
    )
    assert sorted(permutation) == list(range(total_frame_number)), (
        "Permutation does not contain every frame index exactly once."
    )
    return True
