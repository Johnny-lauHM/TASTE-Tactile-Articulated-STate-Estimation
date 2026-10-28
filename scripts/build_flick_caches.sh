#!/bin/bash
# Rebuild the 70 mm finger-trick grasp caches with the swept-volume constraint (holding links stay out of the volume the
# turning layer sweeps): synth (3 seeds, parallel over caches) -> gen_grasp_cache -> verify -> flick_reach -> flick_swept.
#   scripts/build_flick_caches.sh [U D F B]
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sharpa
export OMNI_KIT_ACCEPT_EULA=YES
declare -A C S FREE MV
C[U]="thumb:F:lowr,middle:B:lowr,ring:B:low,index:U:back";      S[U]=U; FREE[U]=index; MV[U]="U,U'"
C[D]="thumb:F:high,middle:B:high,ring:B:lowx:edge,index:U:back"; S[D]=D; FREE[D]=ring;  MV[D]="D,D'"
C[F]="thumb:D:backx,middle:U:backx,index:U:front,ring:U:back";   S[F]=F; FREE[F]=index; MV[F]="F,F'"
C[B]="thumb:D:frontx,middle:U:frontx,index:U:back,ring:U:front"; S[B]=B; FREE[B]=index; MV[B]="B,B'"
# one grip per pair of opposite faces (user: wrists stay put between moves): holding pads on the middle row/column only
# (the pads cannot be aimed off-centre: a 16 mm pad only fits the 23 mm middle row/column when centred; run these with
#  PRESS="0.02 0.05" so the settled grasp does not slide 5 mm away from the synthesized pose)
C[UD]="thumb:F:mid3,middle:B:mid3,index:B:top,ring:B:low";       S[UD]=U,D; FREE[UD]=index,ring; MV[UD]="U,U',D,D'"
C[FB]="thumb:D:midx3,middle:U:midx3,index:U:front,ring:U:back";  S[FB]=F,B; FREE[FB]=index,ring; MV[FB]="F,F',B,B'"
# side pinch for U/D flicks AND the R wrist roll: right hand on the R column only, left hand on the L + middle columns
C[UDR]="thumb:F:mid3,middle:B:mid3,index:B:top,ring:B:low";      S[UDR]=U,D; FREE[UDR]=index,ring; MV[UDR]="U,U',D,D'"
# mirror: left hand on the L column only (L wrist roll), right hand on the R + middle columns
C[UDL]="thumb:F:mid3,middle:B:mid3,index:B:top,ring:B:low";      S[UDL]=U,D; FREE[UDL]=index,ring; MV[UDL]="U,U',D,D'"
declare -A MARGIN; MARGIN[UD]=0.001; MARGIN[FB]=0.001; MARGIN[UDR]=0.001; MARGIN[UDL]=0.001
declare -A RL RR; RL[UDR]=wide; RR[UDR]=edge; RL[UDL]=edge; RR[UDL]=wide  # default: wide / wide
declare -A AX; AX[UDR]="U,D,R"; AX[UDL]="U,D,L"  # internal twist axes to generate (default: the faces of MV); the R roll needs R-axis grasps
BATCH=${BATCH:-4096}
GS=${@:-U D F B}

synth_one() {
  G=$1
  rm -f cache/synth70s_$G.npz  # never let the Isaac stages pick up a stale merged file
  for seed in 0 1 2; do
    python -u scripts/synth_grasps.py --batch $BATCH --seed $seed --region_left ${RL[$G]:-wide} --region_right ${RR[$G]:-wide} --max_normal_cost 0.3 \
      --contacts "${C[$G]}" --swept ${S[$G]} --free ${FREE[$G]} --swept_margin ${MARGIN[$G]:-0.0025} --out cache/synth70s_${G}_$seed.npz 2>&1 | grep -E "^\[synth\]|Traceback|Error" | sed "s/^/[$G s$seed] /"
  done
  python - "$G" <<'EOF'
import glob, sys, numpy as np
G = sys.argv[1]
parts = [dict(np.load(f)) for f in sorted(glob.glob(f"cache/synth70s_{G}_[0-9].npz"))]
parts = [p for p in parts if len(p["loss"])]
out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
np.savez(f"cache/synth70s_{G}.npz", **out)
print(f"[{G}] merged {len(out['loss'])} synthesized grasps -> cache/synth70s_{G}.npz", flush=True)
EOF
}

# two synth jobs at a time (batch 4096 takes ~7 GB of GPU memory each)
set -- $GS
[ -n "$SKIP_SYNTH" ] && set --
while [ $# -gt 0 ]; do
  synth_one $1 > logs/synth70s_$1.log 2>&1 &
  [ -n "$2" ] && synth_one $2 > logs/synth70s_$2.log 2>&1 &
  wait
  shift; [ $# -gt 0 ] && shift
done
cat logs/synth70s_*.log | grep -E "merged|pass rates|satisfy"

for G in $GS; do
  if [ ! -f cache/synth70s_$G.npz ]; then echo "== $G: no synthesized grasps, skipping"; continue; fi
  echo "== $G: gen_grasp_cache"
  AXES=${AX[$G]:-$(echo "${MV[$G]}" | tr -d "'" | tr ',' '\n' | sort -u | paste -sd,)}  # only the internal twist axes this grip is for
  MEM_MAX=34G scripts/run_isolated.sh logs/gen70s_$G.log python -u scripts/gen_grasp_cache.py --headless --num_envs 2048 --synth cache/synth70s_$G.npz --moves "$AXES" \
    --region_left ${RL[$G]:-wide} --region_right ${RR[$G]:-wide} --press ${PRESS:-0.03 0.12} --out cache/g70s_${G}_synth.npz
  grep -E "^\[gen_grasp\]|Traceback" -A3 logs/gen70s_$G.log | tail -8
  echo "== $G: verify"
  MEM_MAX=34G scripts/run_isolated.sh logs/verify70s_$G.log python -u scripts/verify_grasp_cache.py --headless --cache cache/g70s_${G}_synth.npz --out cache/g70s_$G.npz \
    --repeats 2 --region_left ${RL[$G]:-wide} --region_right ${RR[$G]:-wide}
  grep -E "^\[verify\]|Traceback" -A3 logs/verify70s_$G.log | tail -4
  echo "== $G: reach + swept"
  python -u scripts/flick_reach.py --cache cache/g70s_$G.npz --moves "${MV[$G]}" 2>&1 | grep "^\[reach\]"
  python -u scripts/flick_swept.py --cache cache/g70s_$G.npz --moves "${MV[$G]}" --margin ${MARGIN[$G]:-0.0025} 2>&1 | grep "^\[swept\]"
done
echo "[build] done"
# B: the F seeds turned 180 deg about y (scripts/rotate_synth.py); run with SKIP_SYNTH=1 scripts/build_flick_caches.sh B
