"""Budget3D: Idol and Idol-turbo (single image -> 3D Gaussian splats) at any token budget.

The package layout:

    octree, modules, model      the token layouts, the added modules, Idol's denoiser on an adaptive token set
    upstream, encoders, load    Idol's unmodified code, its frozen encoders, loading released and trained weights
    sampling, turbo, fast       the 50-step guided sampler, the 5-step Idol-turbo variant, optimised inference
    curriculum, rollout, rollout_turbo, losses, data, trainer, turbo_parity     training
    splat                       a small Gaussian-splat renderer for galleries and evaluation

Importing ``budget3d`` is cheap; the submodules that need Idol's code import it on first use (see ``upstream``).
"""
__version__ = "1.0.0"
