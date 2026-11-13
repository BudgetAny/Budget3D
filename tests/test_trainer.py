"""CPU check of what the trainers share: the learning-rate schedule, the optimizer groups, and that every recipe in
configs/ is accepted by the trainer it belongs to.

    python tests/test_trainer.py
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch.nn as nn

from budget3d.trainer import lr_scale, param_groups


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.bp_mixer = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 4).requires_grad_(False)


def main():
    assert lr_scale(0, 300, 20000, 0.2) == 1 / 300 and lr_scale(299, 300, 20000, 0.2) == 1.0
    assert abs(lr_scale(300, 300, 20000, 0.2) - 1.0) < 1e-12 and abs(lr_scale(20000, 300, 20000, 0.2) - 0.2) < 1e-12
    assert all(lr_scale(s, 300, 20000, 0.2) >= lr_scale(s + 500, 300, 20000, 0.2) for s in range(300, 19500, 500))

    groups = param_groups(Toy(), 5e-5, 6.0, 1e-4)
    got = sorted((round(g["lr"] * 1e6), g["weight_decay"], sum(p.numel() for p in g["params"])) for g in groups)
    # backbone: weight (decay) | bias + norm (no decay); added module: weight at 6x (decay) | bias (no decay)
    assert got == sorted([(50, 1e-4, 16), (50, 0.0, 4 + 8), (300, 1e-4, 16), (300, 0.0, 4)]), got

    configs = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))
    assert configs, "no recipes in configs/"
    for cfg in configs:
        script = "train_turbo.py" if "turbo" in os.path.basename(cfg) else "train.py"
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), "--config", cfg, "--data", "x",
                            "--print_args"], capture_output=True, text=True)
        assert r.returncode == 0, f"{cfg} is not accepted by {script}:\n{r.stderr[-2000:]}"
        print(f"  {os.path.basename(cfg):28s} -> scripts/{script}: {r.stdout.strip()[:110]}...")
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--config", configs[-1],
                          "--data", "x", "--print_args"], capture_output=True, text=True)
    assert bad.returncode != 0 and "unknown settings" in bad.stderr, "a turbo-only setting must be rejected by train.py"
    print("trainer: schedule, optimizer groups and recipes ok")


if __name__ == "__main__":
    main()
