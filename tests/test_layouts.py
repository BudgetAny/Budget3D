"""CPU check of the token layouts: every budget snaps to an exactly realisable token count N = 512 + 7 k4 + k2, the
layout built from content scores has exactly that many tokens for every example, and it covers each of the 4,096
octree cells exactly once.

    python tests/test_layouts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from budget3d.octree import (N_CELL, N_CUBE, N_SOBOL, build_layout, dense_layout, detail_scores, k4_bounds,
                             snap_realizable_tokens, uniform_layout)


def cells_covered(layout):
    """[B, 4096] how many tokens cover each cell (must be 1 everywhere)."""
    b = layout.batch
    cover = torch.zeros(b, N_CELL)
    cover.scatter_add_(1, layout.idx1, torch.ones_like(layout.idx1, dtype=torch.float))
    cover.scatter_add_(1, layout.idx2, torch.ones_like(layout.idx2, dtype=torch.float))
    # cell index -> the cube it belongs to (cells are indexed on the 16^3 grid, cubes on the 8^3 grid)
    cube_map = torch.arange(N_CUBE).view(8, 1, 8, 1, 8, 1).expand(8, 2, 8, 2, 8, 2).reshape(-1)
    in_cube = torch.zeros(b, N_CUBE)
    in_cube.scatter_add_(1, layout.idx4, torch.ones_like(layout.idx4, dtype=torch.float))
    return cover + in_cube[:, cube_map]


def main():
    assert (N_CUBE, N_CELL, N_SOBOL) == (512, 4096, 8192)
    torch.manual_seed(0)
    z = torch.randn(3, 16, 16, 16, 2, 16) * torch.rand(3, 16, 16, 16, 1, 1)     # uneven detail over space
    s4, s2 = detail_scores(z)
    assert s4.shape == (3, N_CUBE) and s2.shape == (3, N_CELL)
    for budget in (1.0, 0.9, 0.7, 0.5, 0.4, 0.2, 0.1, 0.0625):
        n = snap_realizable_tokens(budget * N_SOBOL)
        k4_bounds(n)                                                             # raises if not realisable
        lay = build_layout(s4, s2, n)
        assert lay.n_tokens == n == N_CUBE + 7 * lay.k4 + lay.k2, (budget, n, lay)
        assert lay.n4 + lay.n2 + lay.n1 == n
        assert bool((cells_covered(lay) == 1).all()), f"budget {budget}: a cell is not covered exactly once"
    assert snap_realizable_tokens(1e9) == N_SOBOL and snap_realizable_tokens(0) == N_CUBE
    assert snap_realizable_tokens(514) == N_CUBE and snap_realizable_tokens(517) == N_CUBE + 7
    dense = dense_layout(2, "cpu")
    assert dense.is_dense and dense.n_tokens == N_SOBOL
    assert uniform_layout(2, 2, "cpu").n_tokens == N_CELL and uniform_layout(2, 4, "cpu").n_tokens == N_CUBE
    print("layouts: exact token counts and full coverage at every budget")


if __name__ == "__main__":
    main()
