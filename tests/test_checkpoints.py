"""CPU check that released weights load and sample with this code's loader (no GPU, a few minutes per checkpoint).

    CKPT=weights/budget3d_idol.safetensors TURBO_CKPT=weights/budget3d_idol_turbo.safetensors python tests/test_checkpoints.py

For each checkpoint given (CKPT: Budget3D-Idol; TURBO_CKPT, TURBO_CKPT_10K: Budget3D-Idol-Turbo):

  * every tensor of the file lands in the model, and every trained parameter of the model is in the file;
  * one forward pass at 10 % of the tokens (819 + the image-condition tokens) on fixed random inputs is finite, and
    the output has Idol's full 8,192-token shape;
  * a full sampling loop at 10 % (2 guided steps for Idol, the 5 guidance-free steps for Idol-turbo) is finite.

Before any trained weights are loaded, the untrained model at 100 % must equal the released Idol exactly (0.0).
With REFERENCE_DIR set, the forward outputs are also compared with tensors saved there (``<name>.pt`` holding
{"latent", "camera"} for the same inputs, see ``fixed_inputs``), e.g. the outputs of an earlier revision of the code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tensordict import TensorDict

from budget3d.load import (check_dense_parity, idol_weights_dir, load_budget3d, load_student, load_teacher,
                           read_checkpoint)
from budget3d.octree import N_SOBOL, build_layout, detail_scores, snap_realizable_tokens
from budget3d.sampling import sample
from budget3d.turbo import sample_turbo, set_lora_fused

BUDGET = 0.1
T_VALUES = (731.0, 200.0)        # one time in each Idol-turbo phase (slow expert above 500, fast below)


def fixed_inputs(batch=1, seed=0):
    """The inputs of the forward check: a latent, a camera token and image-condition features from a seeded generator."""
    g = torch.Generator().manual_seed(seed)
    x = TensorDict({"latent": torch.randn(batch, N_SOBOL, 16, generator=g),
                    "camera": torch.randn(batch, 1, 5, generator=g)}, batch_size=batch)
    cond = TensorDict({"feature1": torch.randn(batch, 4101, 1280, generator=g),
                       "feature2": torch.randn(batch, 4101, 128, generator=g)}, batch_size=batch)
    return x, cond


def forward_at_budget(model, x, cond, t_value, budget=BUDGET):
    z = x["latent"]
    s4, s2 = detail_scores(z[:, model.perm_to_octree].reshape(z.shape[0], 16, 16, 16, 2, -1))
    layout = build_layout(s4, s2, snap_realizable_tokens(budget * N_SOBOL))
    with torch.no_grad():
        out = model(x, torch.full((z.shape[0],), t_value), cond, layout=layout)
    return out, layout


def check(name, path, turbo, weights, ref_dir):
    model, step = load_budget3d(path, turbo=turbo, device="cpu", weights_dir=weights)
    state, _ = read_checkpoint(path)
    trained = {n for n, p in model.named_parameters()}
    assert set(state) <= set(model.state_dict()), "the file has tensors the model does not"
    assert trained <= set(state), f"trained parameters missing from the file: {sorted(trained - set(state))[:5]}"
    sd = model.state_dict()
    assert all(torch.equal(sd[k], v.to(sd[k].dtype)) for k, v in state.items()), "a tensor did not load as stored"
    print(f"[{name}] {os.path.basename(path)}: step {step}, {len(state)} tensors, "
          f"{sum(v.numel() for v in state.values()) / 1e6:.2f} M parameters, all loaded", flush=True)
    x, cond = fixed_inputs()
    for t_value in T_VALUES:
        out, layout = forward_at_budget(model, x, cond, t_value)
        assert out["latent"].shape == (1, N_SOBOL, 16) and out["camera"].shape == (1, 1, 5)
        assert bool(torch.isfinite(out["latent"]).all() and torch.isfinite(out["camera"]).all())
        line = f"[{name}] forward at {BUDGET:.0%} ({layout.n_tokens} tokens), t={t_value:g}: finite"
        ref = os.path.join(ref_dir, f"{name}_t{int(t_value)}.pt") if ref_dir else None
        if ref and os.path.exists(ref):
            want = torch.load(ref)
            d = max(float((out[k] - want[k]).abs().max()) for k in ("latent", "camera"))
            scale = float(want["latent"].abs().max())
            line += f"; max |difference| to the reference {d:.3e} (output scale {scale:.2f})"
            assert d <= float(os.environ.get("REFERENCE_TOL", "0")) * scale, line
        print(line, flush=True)
    if turbo:
        z, cam, layout = sample_turbo(model, cond, BUDGET, steps=5, device="cpu")
    else:
        z, cam, layout = sample(model, cond, BUDGET, steps=2, device="cpu")
    assert z.shape == (1, N_SOBOL, 16) and bool(torch.isfinite(z).all() and torch.isfinite(cam).all())
    print(f"[{name}] sampled at {BUDGET:.0%}: {layout}, latent std {float(z.std()):.3f}", flush=True)


def main():
    torch.manual_seed(0)
    weights = idol_weights_dir()
    todo = [(n, os.environ[e], t) for n, e, t in (("idol", "CKPT", False), ("idol_turbo", "TURBO_CKPT", True),
                                                   ("idol_turbo_10k", "TURBO_CKPT_10K", True)) if os.environ.get(e)]
    if not todo:
        raise SystemExit("set CKPT and / or TURBO_CKPT (TURBO_CKPT_10K) to released weights")
    if not os.environ.get("SKIP_PARITY"):
        rep = check_dense_parity(load_student(weights), load_teacher(weights), batch=1, device="cpu", tol=0.0)
        print(f"[parity] untrained model at 100 % == released Idol: {rep}", flush=True)
    ref_dir = os.environ.get("REFERENCE_DIR")
    if ref_dir:
        set_lora_fused(os.environ.get("REFERENCE_LORA_FUSED", "0") == "1")      # compare in the reference's LoRA form
    for name, path, turbo in todo:
        check(name, path, turbo, weights, ref_dir)
    print("checkpoints: all load and sample")


if __name__ == "__main__":
    main()
