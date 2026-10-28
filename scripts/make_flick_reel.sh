#!/bin/bash
# Record every finger-trick move with its trained policy (one Isaac process per task) and join the clips into one video.
#   scripts/make_flick_reel.sh "Ds:logs/rsl_rl/rubik_flick/<run>/model_700.pt" "Us:..." "Fs:..." "Bs:..."
cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sharpa
export OMNI_KIT_ACCEPT_EULA=YES
OUT=logs/demo/reel
rm -rf $OUT; mkdir -p $OUT
for TC in "$@"; do
  T=${TC%%:*}; C=${TC#*:}
  EYE="-0.40 -0.30 0.46"
  [ "$T" = "Bs" ] && EYE="-0.40 -0.30 0.26"
  MEM_MAX=30G scripts/run_isolated.sh logs/reel_$T.log python -u scripts/flick_reel.py --task Rubik-Flick-$T-Sharpa-v0 --ckpt "$C" --headless --eye $EYE --out_dir $OUT
  grep -E "^\[reel\]|Traceback" -A3 logs/reel_$T.log | grep -E "^\[reel\]|Error"
done
python - <<'EOF'
import glob, os
import imageio.v2 as iio, numpy as np
from PIL import Image, ImageDraw, ImageFont
order = ["U", "Up", "D", "Dp", "F", "Fp", "B", "Bp"]
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 36)
except OSError:
    font = ImageFont.load_default()
w = iio.get_writer("logs/demo/flick_reel.mp4", fps=20)
sheet = []
for m in order:
    p = f"logs/demo/reel/{m}.mp4"
    if not os.path.exists(p):
        continue
    frames = [f for f in iio.get_reader(p)]
    card = Image.new("RGB", (frames[0].shape[1], frames[0].shape[0]), (20, 20, 24))
    ImageDraw.Draw(card).text((40, frames[0].shape[0] // 2 - 24), m.replace("p", "'") + "   (single-finger flick, RL policy, 8x slow motion)", fill=(255, 255, 255), font=font)
    for _ in range(20):
        w.append_data(np.asarray(card))
    for f in frames:
        w.append_data(f)
    sheet.append(np.asarray(Image.open(p.replace(".mp4", "_last.png")).resize((480, 320))))
w.close()
if sheet:
    while len(sheet) % 4:
        sheet.append(np.zeros_like(sheet[0]))
    rows = [np.concatenate(sheet[i:i + 4], 1) for i in range(0, len(sheet), 4)]
    Image.fromarray(np.concatenate(rows, 0)).save("logs/demo/flick_reel_sheet.png")
print("[reel] -> logs/demo/flick_reel.mp4, flick_reel_sheet.png")
EOF
