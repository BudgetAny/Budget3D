"""CPU check: the one-pass (fused) PhaseLoRALinear forward is the classic one, values and gradients.

    python tests/test_lora_fused.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from budget3d import turbo


def run(layer, x, fused):
    turbo.LORA_FUSED = fused
    for p in layer.parameters():
        p.grad = None
    x = x.clone().requires_grad_(True)
    y = layer(x)
    (y * torch.linspace(-1, 1, y.numel()).view_as(y)).sum().backward()
    grads = {n: (torch.zeros_like(p) if p.grad is None else p.grad.clone()) for n, p in layer.named_parameters()}
    return y.detach(), x.grad, grads


def main():
    default = turbo.LORA_FUSED
    torch.manual_seed(0)
    worst = 0.0
    for (i, o) in ((48, 144), (192, 48), (48, 48)):
        st = turbo.PhaseState(allow_uniform_fastpath=True)
        layer = turbo.PhaseLoRALinear(nn.Linear(i, o), 8, 32.0, st)
        for n, p in layer.named_parameters():
            if n.split(".")[-1] in ("slow_A", "slow_B", "fast_A", "fast_B"):
                nn.init.normal_(p, std=0.3)
        cases = [("slow", lambda b: st.set_uniform(True, b, "cpu")), ("fast", lambda b: st.set_uniform(False, b, "cpu")),
                 ("mixed", lambda b: st.set_from_t01(torch.tensor([0.9, 0.1, 0.6, 0.3][:b])))]
        for name, setp in cases:
            for packed in (1, 2):                                    # packed: [rows; rows] with the mask of `rows`
                b = 4
                x = torch.randn(b * packed, 7, i)
                setp(b)
                if name == "mixed":
                    st.uniform = None
                y0, gx0, g0 = run(layer, x, False)
                y1, gx1, g1 = run(layer, x, True)
                d = max(float((y0 - y1).abs().max() / y0.abs().max()), float((gx0 - gx1).abs().max() / gx0.abs().max()),
                        *[float((g0[k] - g1[k]).abs().max() / (g0[k].abs().max() + 1e-12)) for k in g0])
                worst = max(worst, d)
                assert d < 2e-5, (i, o, name, packed, d)
                # an expert that no row uses gets exactly zero gradient, as with where()
                if name == "slow":
                    assert float(g1["fast_A"].abs().max()) == 0.0 and float(g1["fast_B"].abs().max()) == 0.0
    # training configuration: no uniform fast path, every parameter in the graph
    st = turbo.PhaseState(allow_uniform_fastpath=False)
    layer = turbo.PhaseLoRALinear(nn.Linear(32, 64), 8, 32.0, st)
    nn.init.normal_(layer.slow_A, std=0.3), nn.init.normal_(layer.slow_B, std=0.3)
    nn.init.normal_(layer.fast_A, std=0.3), nn.init.normal_(layer.fast_B, std=0.3)
    st.set_from_t01(torch.tensor([0.9, 0.8]))                        # both rows slow, but uniform stays None
    torch.manual_seed(1)
    x = torch.randn(2, 5, 32)
    y0, gx0, g0 = run(layer, x, False)
    y1, gx1, g1 = run(layer, x, True)
    assert st.uniform is None and all(float(g1[k].abs().max()) > 0 for k in ("slow_A", "slow_B", "base.weight"))
    assert float((y0 - y1).abs().max()) < 1e-5 * float(y0.abs().max())
    turbo.LORA_FUSED = default
    print(f"fused LoRA == classic LoRA: worst relative difference {worst:.1e} (values and gradients)")


if __name__ == "__main__":
    main()
