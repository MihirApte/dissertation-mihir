
import sys
import os
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.semantic_shuffle_utils import (
    semantic_permutation,
    cosine_similarity_matrix,
    validate_permutation,
)

# Use numpy arrays - works locally without torch.
# On the professor's GPU machine, torch tensors are used instead
# (cosine_similarity_matrix accepts both).
def _rand_embeddings(n, d=512):
    e = np.random.randn(n, d).astype(np.float32)
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    return e

PASS = "  [PASS]"
FAIL = "  [FAIL]"

def test_cosine_similarity_matrix():
    """Similarity matrix should be symmetric and have 1s on diagonal."""
    embeddings = _rand_embeddings(10)
    sim = cosine_similarity_matrix(embeddings)

    assert sim.shape == (10, 10), f"Expected (10,10), got {sim.shape}"
    assert np.allclose(sim, sim.T, atol=1e-5), "Similarity matrix not symmetric"
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5), "Diagonal should be 1.0"
    print(PASS, "cosine_similarity_matrix: shape, symmetry, diagonal")

def test_permutation_is_valid():
    """Output must be a valid permutation of all frame indices."""
    total = 90       # 10 grids x 9 frames
    grid_k = 9
    embeddings = _rand_embeddings(total)

    perm = semantic_permutation(embeddings, grid_k, total)
    validate_permutation(perm, total)
    print(PASS, "semantic_permutation: valid permutation of 90 frames")

def test_permutation_correct_length():
    """Works for different grid sizes."""
    for grid_k in [4, 9, 16]:
        total = grid_k * 8
        embeddings = _rand_embeddings(total)
        perm = semantic_permutation(embeddings, grid_k, total)
        assert len(perm) == total
        assert sorted(perm) == list(range(total))
    print(PASS, "semantic_permutation: correct length for grid sizes 4, 9, 16")

def test_similar_frames_grouped():
    """
    Frames that are nearly identical should end up in the same group.
    We create 3 clusters of 3 identical frames each (9 frames total,
    grid_k=3). Semantic shuffle should group each cluster together.
    """
    total = 9
    grid_k = 3

    # 3 orthogonal cluster directions, padded to 512 dims
    base = np.zeros((3, 512), dtype=np.float32)
    base[0, 0] = 1.0   # cluster A
    base[1, 1] = 1.0   # cluster B
    base[2, 2] = 1.0   # cluster C

    # 3 copies of each: frames 0,1,2 -> A; 3,4,5 -> B; 6,7,8 -> C
    embeddings = np.concatenate([base[0:1]] * 3 +
                                [base[1:2]] * 3 +
                                [base[2:3]] * 3, axis=0)

    # Run many times - clusters should always stay together
    all_grouped = True
    for _ in range(20):
        perm = semantic_permutation(embeddings, grid_k, total)
        for start in range(0, total, grid_k):
            group = set(perm[start:start + grid_k])
            in_a = group <= {0, 1, 2}
            in_b = group <= {3, 4, 5}
            in_c = group <= {6, 7, 8}
            if not (in_a or in_b or in_c):
                all_grouped = False
                break

    assert all_grouped, "Similar frames not consistently grouped together"
    print(PASS, "semantic_permutation: similar frames grouped into same grid")

def test_different_calls_differ():
    """
    Two consecutive calls should (almost always) produce different
    permutations - so different denoising steps get different groupings.
    """
    total = 90
    grid_k = 9
    embeddings = _rand_embeddings(total)

    perm1 = semantic_permutation(embeddings, grid_k, total)
    perm2 = semantic_permutation(embeddings, grid_k, total)

    # With 90 frames it's astronomically unlikely to get identical perms
    assert perm1 != perm2, "Two consecutive calls returned identical permutations"
    print(PASS, "semantic_permutation: different calls produce different orderings")

def test_validate_permutation_catches_errors():
    """validate_permutation should raise on bad input."""
    try:
        validate_permutation([0, 1, 2, 2], 4)  # duplicate index
        print(FAIL, "validate_permutation: should have raised on duplicate")
        return False
    except AssertionError:
        pass

    try:
        validate_permutation([0, 1, 2], 4)  # wrong length
        print(FAIL, "validate_permutation: should have raised on wrong length")
        return False
    except AssertionError:
        pass

    print(PASS, "validate_permutation: correctly catches invalid permutations")


if __name__ == '__main__':
    print("\n-- Semantic Shuffle Logic Tests --------------------------")
    passed = 0
    failed = 0

    tests = [
        test_cosine_similarity_matrix,
        test_permutation_is_valid,
        test_permutation_correct_length,
        test_similar_frames_grouped,
        test_different_calls_differ,
        test_validate_permutation_catches_errors,
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
