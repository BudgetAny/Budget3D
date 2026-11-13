"""Pre-flight checks on a real GPU: parity with the released model, the adapter inventory,
the encoders on a real render, and a few real training steps across the budget range."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tensordict import TensorDict

from budget3d.curriculum import LayoutSampler
from budget3d.data import image_stream
from budget3d.encoders import build_encoder
from budget3d.load import check_dense_parity, idol_weights_dir, load_student, load_teacher
from budget3d.losses import distill_loss
from budget3d.octree import build_layout, detail_scores, point_scales, uniform_layout
from budget3d.rollout import RolloutStream

REF_ADAPTERS = {"bp_scale_emb": 4096, "bp_ipe": 288000, "bp_mixers.2": 791296, "bp_mixers.4": 2387200,
                "bp_refiners.4": 1980688, "bp_refiners.2": 1979920, "bp_feat_proj": 132224}


def main():
    dev = "cuda"
    data = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DOME_ROOT", "data/dome_objaverse")
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0),
          "| %.0f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30), flush=True)

    weights = idol_weights_dir()
    student = load_student(weights, device=dev)
    teacher_ref = load_teacher(weights, device=dev)

    print("\n== 1. adapter inventory vs the reference table ==", flush=True)
    got, bad = {}, 0
    for name, _ in REF_ADAPTERS.items():
        got[name] = sum(p.numel() for n, p in student.named_parameters() if n.startswith(name))
        d = got[name] - REF_ADAPTERS[name]
        bad += d != 0
        print(f"  {name:16s} ref {REF_ADAPTERS[name]:>9,d}  got {got[name]:>9,d}  {'OK' if d == 0 else 'MISMATCH'}")
    n_bp = sum(p.numel() for p in student.bp_parameters)
    n_all = sum(p.numel() for p in student.parameters())
    print(f"  total adapters {n_bp:,d} (ref 7,563,424) | whole model {n_all/1e6:.2f}M (ref 378.11M) | "
          f"backbone {(n_all-n_bp)/1e6:.2f}M (ref 370.55M)")
    assert bad == 0 and n_bp == 7_563_424, "adapter inventory does not match the specification"

    print("\n== 2. dense parity against the released Idol denoiser (fp32, exact) ==", flush=True)
    rep = check_dense_parity(student, teacher_ref, batch=1, device=dev, tol=0.0)
    print("  max |student - released| =", rep, "-> bit-exact", flush=True)
    del teacher_ref
    torch.cuda.empty_cache()

    print("\n== 3. frozen encoders on a real Dome-Objaverse render ==", flush=True)
    enc = build_encoder(device=dev)
    imgs = image_stream(data, 2, num_workers=0)
    batch = next(imgs).to(dev)
    t = time.time()
    feats = enc(batch)
    print(f"  image {tuple(batch.shape)} -> feature1 {tuple(feats['feature1'].shape)} "
          f"feature2 {tuple(feats['feature2'].shape)} in {time.time()-t:.2f}s")
    assert feats["feature1"].shape[1:] == (4101, 1280) and feats["feature2"].shape[1:] == (4101, 128)

    print("\n== 4. teacher rollout + student step across the budget range ==", flush=True)
    teacher = load_student(weights, device=dev).eval().requires_grad_(False)
    student.grad_checkpointing = True
    student.train()
    stream = RolloutStream(teacher, enc, imgs, n_traj=2, device=dev, seed=0)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-6)
    idx = torch.arange(2, device=dev)
    for n_tokens in [8192, 4915, 4096, 1638, 819, 512]:
        tg = stream.targets(idx)
        z_oct = tg["x"]["latent"][:, student.perm_to_octree].reshape(2, 16, 16, 16, 2, -1)
        s4, s2 = detail_scores(z_oct)
        layout = build_layout(s4, s2, n_tokens)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        opt.zero_grad(set_to_none=True)
        if True:
            out = student(tg["x"], tg["t01"] * 1000, TensorDict(
                {"feature1": stream.f1[idx].float(), "feature2": stream.f2[idx].float()}, batch_size=2),
                layout=layout, return_features=True)
            loss, parts = distill_loss(out, tg, point_scales(layout, student.inv_perm))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        opt.step()
        torch.cuda.synchronize()
        assert out["latent"].shape == (2, 8192, 16) and torch.isfinite(out["latent"]).all()
        grads = sum(1 for p in student.bp_parameters if p.grad is not None and torch.isfinite(p.grad).all())
        print(f"  N={n_tokens:5d} loss {parts["total"]:.3e} (vel {parts["vel"]:.3e} cam {parts["cam"]:.3e} "
              f"feat {parts['feat_nmse']:.3f}/{parts['feat_cos']:.3f})  gnorm {float(gn):5.2f}  "
              f"{time.time()-t:.2f}s  peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB  "
              f"bp grads finite {grads}/{len(student.bp_parameters)}", flush=True)
        stream.advance(idx, tg)

    print("\n== 5. uniform-coarse layouts (every adapter must receive gradient) ==", flush=True)
    for scale in (2, 4):
        tg = stream.targets(idx)
        layout = uniform_layout(2, scale, dev)
        opt.zero_grad(set_to_none=True)
        if True:
            out = student(tg["x"], tg["t01"] * 1000, TensorDict(
                {"feature1": stream.f1[idx].float(), "feature2": stream.f2[idx].float()}, batch_size=2),
                layout=layout, return_features=True)
            loss, parts = distill_loss(out, tg, point_scales(layout, student.inv_perm))
        loss.backward()
        touched = sum(1 for p in student.bp_parameters if p.grad is not None and float(p.grad.abs().sum()) > 0)
        print(f"  uniform s={scale} (N={layout.n_tokens}) loss {parts['total']:.3f}  "
              f"adapters with gradient {touched}/{len(student.bp_parameters)}", flush=True)
        stream.advance(idx, tg)

    print("\n== 6. every curriculum layout covers all 8192 points ==", flush=True)
    ls = LayoutSampler(device=dev, seed=11)
    import collections
    cover = collections.Counter()
    for _ in range(60):
        z_oct = torch.randn(2, 16, 16, 16, 2, 16, device=dev)
        layout, mode = ls(z_oct, torch.tensor([0, 3], device=dev))
        ps = point_scales(layout, student.inv_perm)
        assert ps.shape == (2, 8192)
        assert bool((ps > 0).all()), f"{mode} {layout} left Sobol points unwritten"
        assert bool(((ps == 1) | (ps == 2) | (ps == 4)).all())
        cover[mode] += 1
    print("  60 layouts, all 8192 points owned by exactly one token in every one:", dict(cover), flush=True)

    print("\n== 7. layout curriculum, 12 draws ==", flush=True)
    ls = LayoutSampler(device=dev, seed=3)
    for i in range(12):
        tg = stream.targets(idx)
        z_ref = ls.reference_latent(tg["x"]["latent"], stream.z0[idx], stream.k[idx])
        layout, mode = ls(z_ref[:, student.perm_to_octree].reshape(2, 16, 16, 16, 2, -1), stream.k[idx])
        print(f"  k={int(stream.k[idx][0]):2d} t={float(tg['t01'][0]):.3f} mode {mode:9s} {layout}", flush=True)
        stream.advance(idx, tg)
    print("\nSMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
