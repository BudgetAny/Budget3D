#!/usr/bin/env bash
# The Budget3D tests that need no GPU, in order; stops at the first failure.
#   bash tests/run_cpu.sh                      # unit tests: no weights needed (Idol's code must be importable)
#   CKPT=weights/budget3d_idol.safetensors TURBO_CKPT=weights/budget3d_idol_turbo.safetensors bash tests/run_cpu.sh
#                                              # ... and the released weights: load, forward and sample on CPU (~64 GB RAM)
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
step() { echo; echo "=== $1"; }
step "1/5 token layouts"; $PY tests/test_layouts.py
step "2/5 one-pass phase-LoRA == classic phase-LoRA (values and gradients)"; $PY tests/test_lora_fused.py
step "3/5 schedule, optimizer groups, recipes"; $PY tests/test_trainer.py
step "4/5 every script imports and parses"; $PY tests/test_scripts.py
if [ -n "$CKPT$TURBO_CKPT$TURBO_CKPT_10K" ]; then
  step "5/5 released weights: load, forward, sample (CPU)"; $PY tests/test_checkpoints.py
else
  step "5/5 released weights: skipped (set CKPT / TURBO_CKPT)"
fi
echo; echo "all Budget3D CPU tests passed"
