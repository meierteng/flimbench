"""Contact sheets (rows = style, cols = seed x {start, mid, end}) and simple per-clip stats.
python analyze.py DIR"""
import sys, os, glob, json, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
D = sys.argv[1]
ORDER = ["none", "nolan", "burton", "kubrick", "wes_anderson", "zhang_yimou", "spielberg"]

def frames(path, n=3):
    cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for i in np.linspace(0, N - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if ok: out.append(f)
    return out

def stats(path):
    cap = cv2.VideoCapture(path); fs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fs.append(f)
    fs = fs[::2]
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32) for f in fs]
    lab = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in fs]
    L = np.concatenate([x[..., 0].ravel() for x in lab]) * 100 / 255
    b = np.concatenate([x[..., 2].ravel() for x in lab]) - 128   # + yellow/warm, - blue/cool
    a_ = np.concatenate([x[..., 1].ravel() for x in lab]) - 128  # + magenta/red, - green
    S = np.concatenate([x[..., 1].ravel() for x in hsv]) / 255
    g = [cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fs]
    sym = np.mean([1 - np.abs(x - x[:, ::-1]).mean() / 255 for x in g])
    flow = np.mean([np.linalg.norm(cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, .5, 3, 15, 3, 5, 1.2, 0), axis=2).mean()
                    for i in range(len(g) - 1)])
    return dict(luma=L.mean(), contrast=np.percentile(L, 95) - np.percentile(L, 5), sat=S.mean(),
                warm_b=b.mean(), red_a=a_.mean(), symmetry=sym, motion=flow)

clips = sorted(glob.glob(os.path.join(D, "*__diner__*.mp4")))
rows = []
for p in clips:
    m, c, s, sd = os.path.basename(p)[:-4].split("__")
    rows.append(dict(model=m, style=s, seed=sd, **{k: round(float(v), 3) for k, v in stats(p).items()}))
json.dump(rows, open(os.path.join(D, "stats.json"), "w"), indent=1)

try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
except Exception: font = ImageFont.load_default()
W, H, LW = 320, 180, 170
for m in sorted({r["model"] for r in rows}):
    seeds = sorted({r["seed"] for r in rows if r["model"] == m})
    sheet = Image.new("RGB", (LW + W * 3 * len(seeds) + 10 * len(seeds), H * len(ORDER)), "white")
    d = ImageDraw.Draw(sheet)
    for i, s in enumerate(ORDER):
        d.text((8, i * H + H // 2 - 12), s, fill="black", font=font)
        for j, sd in enumerate(seeds):
            p = os.path.join(D, f"{m}__diner__{s}__{sd}.mp4")
            if not os.path.exists(p): continue
            for k, f in enumerate(frames(p)):
                im = Image.fromarray(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB))
                sheet.paste(im, (LW + j * (3 * W + 10) + k * W, i * H))
    sheet.save(os.path.join(D, f"sheet_{m}.jpg"), quality=88)

import collections
for m in sorted({r["model"] for r in rows}):
    print(f"\n== {m}")
    keys = ["luma", "contrast", "sat", "warm_b", "red_a", "symmetry", "motion"]
    print(f"{'style':14s}" + "".join(f"{k:>10s}" for k in keys))
    for s in ORDER:
        rs = [r for r in rows if r["model"] == m and r["style"] == s]
        if rs: print(f"{s:14s}" + "".join(f"{np.mean([r[k] for r in rs]):10.3f}" for k in keys))
