#!/usr/bin/env bash
# Every Budget3D test, in order, on one GPU (about 20 minutes); stops at the first failure.
#   CKPT        Budget3D-Idol weights (weights/budget3d_idol.safetensors)
#   TURBO_CKPT  optional: Budget3D-Idol-Turbo weights (a .safetensors or a training checkpoint); its sampling is tested too
#   DOME_ROOT   a Dome-Objaverse subset (scripts/fetch_dome.py), for the training steps
#   OUT         where the test outputs go (runs/tests)
#   CKPT=weights/budget3d_idol.safetensors DOME_ROOT=data/dome_objaverse bash tests/run_gpu.sh
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${OUT:-runs/tests}
D=${DOME_ROOT:-data/dome_objaverse}
mkdir -p "$OUT"
EXAMPLES=${IDOL_ROOT:-third_party/idol-sdk}/demo/examples
IMG1=$EXAMPLES/fig-knight.webp
IMG2=$EXAMPLES/food-donut.webp
step() { echo; echo "=== $1"; }
step "1/8 the CPU unit tests"; SKIP_PARITY=1 CKPT= TURBO_CKPT= TURBO_CKPT_10K= bash tests/run_cpu.sh
step "2/8 parity with the released Idol, adapter inventory, encoders, training steps"; $PY tests/gpu_smoke.py "$D"
step "3/8 idol-turbo: parity with the official turbo code, the sampler, budgets"; $PY tests/gpu_smoke_turbo.py --image "$IMG1" --out "$OUT/turbo"
step "4/8 sampling from the released weights at 100 % and 50 %"
for b in 1.0 0.5; do $PY scripts/sample.py --ckpt "$CKPT" --image "$IMG1" --budget $b --out "$OUT/knight_b$b.ply"; done
if [ -n "$TURBO_CKPT" ]; then
  $PY scripts/sample.py --ckpt "$TURBO_CKPT" --turbo --image "$IMG1" --budget 0.5 --out "$OUT/knight_turbo_b0.5.ply"
fi
step "5/8 gallery: two objects at two budgets, rendered orbits"
$PY scripts/gallery.py --ckpt "$CKPT" --out "$OUT/gallery" --images "$IMG1" "$IMG2" --budgets 1.0 0.5 --batch 2
step "6/8 MP4 export"; $PY scripts/to_mp4.py --out "$OUT/gallery"
step "7/8 quality against budget"; $PY scripts/eval_budget.py --gallery "$OUT/gallery" --out "$OUT/eval.json" --views 4
step "8/8 both trainers from their configs: 2 steps, a checkpoint, then a resume to step 3 (one GPU)"
TINY="--batch_per_gpu 1 --grad_accum 1 --num_workers 2 --log_every 1 --save_every 1 --snapshot_every 1000"
for t in "train configs/idol.yaml" "train_turbo configs/idol_turbo.yaml"; do
  set -- $t; rm -rf "$OUT/$1"
  $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) scripts/$1.py --config "$2" --data "$D" --out "$OUT/$1" $TINY --max_steps 2
  $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$((20000 + RANDOM % 20000)) scripts/$1.py --config "$2" --data "$D" --out "$OUT/$1" $TINY --max_steps 3 --resume --skip_parity
  ls "$OUT/$1"/checkpoint-last.pth > /dev/null
done
echo; echo "all Budget3D tests passed"
