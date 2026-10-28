import sys, imageio.v2 as imageio
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
idx = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
fr = [f for f in imageio.get_reader(src)]
idx = idx or list(range(0, len(fr), max(1, len(fr) // 12)))[:12]
idx = [i for i in idx if i < len(fr)]
ims = [Image.fromarray(fr[i]).resize((320, 240)) for i in idx]
W = Image.new("RGB", (320 * 4, 240 * ((len(ims) + 3) // 4)))
for k, im in enumerate(ims):
    W.paste(im, ((k % 4) * 320, (k // 4) * 240))
W.save(dst)
print(len(fr), idx)
