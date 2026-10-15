"""Real CMD clips per director: letterbox crop, shot split, sample frames, same stats as analyze.py.
python refs.py REFDIR GENDIR"""
import sys, os, json, glob, cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
R, G = sys.argv[1], sys.argv[2]
pk = pd.read_csv(os.path.join(R, "picked.csv"))
ORDER = ["nolan", "burton", "kubrick", "wes_anderson", "zhang_yimou", "spielberg"]
LABEL = {"none": "No style", "nolan": "Nolan", "burton": "Burton", "kubrick": "Kubrick",
         "wes_anderson": "Wes Anderson", "zhang_yimou": "Zhang Yimou", "spielberg": "Spielberg"}
TRAITS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "traits.json")))

def read(path, step):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 24; fs = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0: fs.append(f)
        i += 1
    return fs, fps / step

def crop_box(fs):
    g = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fs[:: max(1, len(fs) // 40)]], 0)
    rows = np.where(g.mean(1) > 18)[0]; cols = np.where(g.mean(0) > 18)[0]
    if len(rows) < 10 or len(cols) < 10: return 0, g.shape[0], 0, g.shape[1]
    return rows[0] + 2, rows[-1] - 1, cols[0] + 2, cols[-1] - 1

def shots(fs):
    h = [cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]) for f in fs]
    h = [cv2.normalize(x, x).flatten() for x in h]
    cut = [0] + [i for i in range(1, len(h)) if cv2.compareHist(h[i - 1], h[i], cv2.HISTCMP_BHATTACHARYYA) > 0.35] + [len(fs)]
    return [(a, b) for a, b in zip(cut[:-1], cut[1:]) if b - a >= 2]

def stats(fs, segs):
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32) for f in fs]
    lab = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in fs]
    L = np.concatenate([x[..., 0].ravel() for x in lab]) * 100 / 255
    g = [cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fs]
    flows = [np.linalg.norm(cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, .5, 3, 15, 3, 5, 1.2, 0), axis=2).mean()
             for a, b in segs for i in range(a, b - 1)]
    return dict(luma=L.mean(), contrast=np.percentile(L, 95) - np.percentile(L, 5),
                sat=np.mean([x[..., 1].mean() for x in hsv]) / 255,
                warm_b=np.mean([x[..., 2].mean() for x in lab]) - 128, red_a=np.mean([x[..., 1].mean() for x in lab]) - 128,
                symmetry=np.mean([1 - np.abs(x - x[:, ::-1]).mean() / 255 for x in g]),
                motion=float(np.mean(flows)) if flows else np.nan)

rows, thumbs = [], {s: [] for s in ORDER}
for _, r in pk.iterrows():
    p = glob.glob(os.path.join(R, r.videoid + ".*"))
    p = [x for x in p if x.endswith((".mp4", ".webm", ".mkv"))]
    if not p: continue
    fs, efps = read(p[0], 1)
    step = max(1, round(efps / 5)); fs = fs[::step]; efps /= step           # ~5 frames per second
    fs = fs[int(2 * efps): len(fs) - int(3 * efps)]                         # drop channel intro and outro
    def endcard(f):                                                         # MovieClips end screen: flat dark-gray UI
        h = cv2.cvtColor(f[:, f.shape[1] // 2:], cv2.COLOR_BGR2HSV)
        return np.mean((h[..., 1] < 25) & (h[..., 2] > 25) & (h[..., 2] < 75)) > 0.45
    tail0 = max(0, len(fs) - int(60 * efps))
    ec = [i for i in range(tail0, len(fs)) if endcard(fs[i])]
    if ec: fs = fs[: ec[0]]
    t, b, l, rr = crop_box(fs); fs = [f[t:b, l:rr] for f in fs]
    fs = [cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))) for f in fs]
    sg = shots(fs)
    st = stats(fs, sg)
    rows.append(dict(style=r["style"], videoid=r.videoid, title=r.title, n_shots=len(sg),
                     asl=round(len(fs) / efps / max(1, len(sg)), 2), **{k: round(float(v), 3) for k, v in st.items()}))
    longest = sorted(sg, key=lambda x: x[1] - x[0], reverse=True)[:2]
    for a, b in sorted(longest): thumbs[r["style"]].append((fs[(a + b) // 2], r.title))
json.dump(rows, open(os.path.join(R, "ref_stats.json"), "w"), indent=1, ensure_ascii=False)
print(pd.DataFrame(rows).groupby("style")[["n_shots", "asl", "luma", "sat", "warm_b", "contrast", "motion", "symmetry"]].agg(["mean", "std"]).round(2).to_string())

# figure 1: real frames vs generated mid frames
TW, TH_, LW = 224, 126, 230
fT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
fS = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 16)
fN = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 12)
ncr, gap = 4, 16
RH = TH_ * 2
W = LW + TW * ncr + gap + TW * 2; H = 40 + RH * len(ORDER) + 8 * len(ORDER)
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
d.text((LW + 8, 10), "真实影片（CMD 片段，不同影片的不同镜头）", fill="black", font=fS)
d.text((LW + TW * ncr + gap + 8, 10), "生成（Wan 上 / Veo 下，种子 1、2）", fill="black", font=fS)
def mid(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, f = cap.read(); return f if ok else None
for i, s in enumerate(ORDER):
    y = 40 + i * (RH + 8)
    d.text((10, y + 8), LABEL[s], fill="black", font=fT)
    for k, t in enumerate(TRAITS[s]): d.text((10, y + 40 + k * 22), t, fill=(70, 70, 70), font=fS)
    films = sorted({t for _, t in thumbs[s]})
    d.text((10, y + 40 + 3 * 22 + 8), "；".join(films)[:60], fill=(120, 120, 120), font=fN)
    for j, (f, _) in enumerate(thumbs[s][: ncr * 2]):
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); im.thumbnail((TW, TH_))
        cell = Image.new("RGB", (TW, TH_), "black"); cell.paste(im, ((TW - im.width) // 2, (TH_ - im.height) // 2))
        img.paste(cell, (LW + (j % ncr) * TW, y + (j // ncr) * TH_))
    for r_, m in enumerate(["wan-3.0", "veo-3.1-lite"]):
        for c_, sd in enumerate(["s1", "s2"]):
            p = os.path.join(G, f"{m}__diner__{s}__{sd}.mp4")
            f = mid(p) if os.path.exists(p) else None
            cell = Image.new("RGB", (TW, TH_), (235, 235, 235))
            if f is not None:
                im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); im.thumbnail((TW, TH_)); cell.paste(im, (0, 0))
            else:
                ImageDraw.Draw(cell).text((10, TH_ // 2 - 8), "被平台拦截", fill=(80, 80, 80), font=fS)
            img.paste(cell, (LW + TW * ncr + gap + c_ * TW, y + r_ * TH_))
img.save(os.path.join(G, "real_vs_generated.jpg"), quality=88)

# figure 2: metrics with real distribution
gen = json.load(open(os.path.join(G, "stats.json")))
mets = [("luma", "Brightness (L*)"), ("sat", "Saturation"), ("warm_b", "Warm (+) / cool (-), b*"),
        ("contrast", "Contrast (L* p95 - p5)"), ("red_a", "Red (+) / green (-), a*"), ("symmetry", "Left-right symmetry")]
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5)); xs = np.arange(len(ORDER))
srcs = [("real", "#55A868", "Real films (each dot = one clip)"), ("wan-3.0", "#4C72B0", "Wan 3.0"), ("veo-3.1-lite", "#DD8452", "Veo 3.1 Lite")]
for ax, (k, title) in zip(axs.ravel(), mets):
    for o, (src, c, nm) in enumerate(srcs):
        vals = [[r[k] for r in (rows if src == "real" else gen) if r["style"] == s and (src == "real" or r["model"] == src)] for s in ORDER]
        mu = [np.nanmean(v) if v else np.nan for v in vals]; xo = xs + (o - 1) * .27
        ax.bar(xo, mu, .27, color=c, label=nm, alpha=.9)
        for xi, v in zip(xo, vals): ax.scatter([xi] * len(v), v, s=9, color="black", zorder=3)
    ax.set_title(title); ax.set_xticks(xs); ax.set_xticklabels([LABEL[s] for s in ORDER], rotation=30, ha="right", fontsize=9)
axs[0, 0].legend(fontsize=8)
fig.suptitle("Real director clips (CMD, 3-4 films x 2 clips) vs generated diner clips (2 seeds). Real metrics are computed over whole clips; generated clips are a single 4-5 s shot.", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(G, "real_vs_generated_metrics.png"), dpi=130)
print("ok")
