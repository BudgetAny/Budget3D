"""GPU smoke test for the Idol-turbo variant (run before a training job).

  1. parity: our per-row phase-LoRA student == official turbo.py on Idol's class (slow, fast, mixed batch)
  2. sampler: sample_turbo at 100 % == the official guidance-free phase-switched Euler loop (same noise, real image)
  3. budgets: the untrained student samples at 70 % / 40 % (runs; quality is expected to be poor before training)
  4. renders one view of each into --out for a visual check
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    from gallery import Preprocessor, decode, object_noise, orbit_camera
    from budget3d.load import idol_weights_dir, load_student, load_teacher
    from budget3d.sampling import time_grid
    from budget3d.splat import Splats, orbit_positions, render
    from budget3d.turbo import lora_path, make_turbo, sample_turbo, set_lora_fused
    from budget3d.turbo_parity import check_turbo_parity, official_turbo_module

    weights = idol_weights_dir()
    lora = lora_path()
    # 1. parity (training configuration: both experts evaluated, selected per row)
    st_train = load_student(weights, device=dev)
    make_turbo(st_train, lora, allow_uniform_fastpath=False)
    print("[smoke] parity:", check_turbo_parity(st_train, weights, lora, device=dev), flush=True)
    del st_train
    torch.cuda.empty_cache()

    # 2. sampler parity at 100 %, fp32, real conditioning; in the classic LoRA form, which is the official code's
    was_fused = set_lora_fused(False)
    from budget3d.load import load_decoders
    dec, dec_gs, cfg_json = load_decoders(weights, dev)
    prep = Preprocessor(weights, dev, "synthetic")
    _, cond = prep(a.image)
    student = load_student(weights, device=dev)
    make_turbo(student, lora)
    student.bind_geometry().eval()
    for m in student.modules():
        if hasattr(m, "bf16_blocks"):
            m.bf16_blocks = False
    z1, c1 = object_noise(student, 42, dev)
    lat_s, cam_s, _ = sample_turbo(student, cond, 1.0, a.steps, noise=(z1.clone(), c1.clone()))

    off = official_turbo_module()
    ref = load_teacher(weights, device=dev)
    off._inject(ref, 32, 128.0)
    ref.to(dev)
    from safetensors.torch import load_file
    ref.load_state_dict(load_file(lora), strict=False)
    z, cam = z1.clone(), c1.clone()
    grid = time_grid(a.steps, 3.0)
    with torch.no_grad():
        for i in range(a.steps):
            t, tp = float(grid[i]), float(grid[i + 1])
            off._set_active(ref, "slow" if t >= 0.5 else "fast")
            out = ref(TensorDict({"latent": z, "camera": cam}, batch_size=1), torch.tensor([1000 * t], device=dev), cond)
            z = z - (t - tp) * out["latent"]
            cam = cam - (t - tp) * out["camera"]
    err = float((lat_s - z).norm() / z.norm())
    print(f"[smoke] sampler parity at 100%: rel L2 {err:.2e} (camera max diff {(cam_s - cam).abs().max().item():.2e})", flush=True)
    assert err < 1e-4, err
    set_lora_fused(was_fused)
    del ref
    torch.cuda.empty_cache()

    # 3./4. budgets with the untrained student, and renders
    tiles = []
    for b, (lat, camv) in [(1.0, (lat_s, cam_s))] + [
            (b, sample_turbo(student, cond, b, a.steps, noise=(z1.clone(), c1.clone()))[:2]) for b in (0.7, 0.4)]:
        c = orbit_camera(camv[0, 0])
        sp = Splats(decode(lat.float(), dec, dec_gs, cfg_json, 262144))
        r = float(torch.quantile(sp.xyz.norm(dim=1), 0.995)) / math.sin(math.radians(20))
        for k in (0, 1):
            tiles.append((render(sp, orbit_positions(c["yaw"], c["pitch"], r, 4)[k], 40, size=320).cpu().numpy()
                          * 255).round().astype(np.uint8))
        print(f"[smoke] sampled {b:.0%}", flush=True)
    Image.fromarray(np.concatenate(tiles, 1)).save(os.path.join(a.out, "turbo_views.png"))
    print("[smoke] ok", flush=True)


if __name__ == "__main__":
    main()
