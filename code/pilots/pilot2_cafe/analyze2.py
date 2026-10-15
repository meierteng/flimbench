"""Contact sheet (rows = style, column groups = condition x model, 4 frames each), cut count, per-shot metrics.
python analyze2.py GENDIR"""
import sys, os, json, glob, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from trait_metrics import shot_metrics
D = sys.argv[1]
ORDER = ["none", "nolan", "burton", "kubrick", "wes_anderson", "zhang_yimou", "spielberg"]
LABEL = {"none": "No style", "nolan": "Nolan", "burton": "Burton", "kubrick": "Kubrick", "wes_anderson": "Wes Anderson",
         "zhang_yimou": "Zhang Yimou", "spielberg": "Spielberg"}
GROUPS = [("", "wan-3.0", "Name prompt: Wan 3.0"), ("", "veo-3.1-lite", "Name prompt: Veo 3.1 Lite"),
          ("desc_", "wan-3.0", "Description prompt: Wan 3.0"), ("desc_", "veo-3.1-lite", "Description prompt: Veo 3.1 Lite")]
STY = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles_gpt54.json")))["styles"]

def load(p):
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 24; fs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fs.append(f)
    return fs, fps

def cuts(fs):
    def h(f):
        a = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
        b = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(a, a).flatten(), cv2.normalize(b, b).flatten()
    H = [h(f) for f in fs]
    return [i for i in range(1, len(H)) if max(cv2.compareHist(H[i-1][0], H[i][0], 3), cv2.compareHist(H[i-1][1], H[i][1], 3)) > 0.33]

rows = []
for p in sorted(glob.glob(os.path.join(D, "*__cafe__*.mp4"))):
    m, c, s, sd = os.path.basename(p)[:-4].split("__")
    cond = "desc" if s.startswith("desc_") else ("none" if s == "none" else "name")
    style = s[5:] if cond == "desc" else s
    fs, fps = load(p); cs = cuts(fs)
    segs = list(zip([0] + cs, cs + [len(fs)]))
    step = max(1, round(fps / 6)); per = []
    for a, b in segs:
        sub = fs[a:b:step]
        if len(sub) >= 3: per.append(shot_metrics(sub, fps / step))
    agg = {k: float(np.nanmean([x[k] for x in per])) for k in per[0]} if per else {}
    rows.append(dict(model=m, cond=cond, style=style, seed=sd, n_shots=len(segs), cut_times=[round(i / fps, 2) for i in cs],
                     duration=round(len(fs) / fps, 2), **{k: round(v, 3) for k, v in agg.items() if k != "shot_seconds"}))
json.dump(rows, open(os.path.join(D, "metrics.json"), "w"), indent=1)

TW, TH, LW, HH, GAP = 192, 108, 300, 64, 14
W = LW + len(GROUPS) * (4 * TW + GAP); H = HH + len(ORDER) * (TH + 8)
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
A = "/System/Library/Fonts/Supplemental/Arial"
fB, fH, fS = ImageFont.truetype(A + " Bold.ttf", 17), ImageFont.truetype(A + " Bold.ttf", 15), ImageFont.truetype(A + ".ttf", 11)
d.text((10, 8), "Content: talk over coffee, then stand up and leave", fill="black", font=fH)
d.text((10, 30), "Seed 1, 8 s. Frames at 10, 40, 70, 95% of the clip.", fill=(110, 110, 110), font=fS)
d.text((10, 44), "Red bar = a cut detected between frames.", fill=(180, 0, 0), font=fS)
for g, (_, _, t) in enumerate(GROUPS): d.text((LW + g * (4 * TW + GAP) + 6, 38), t, fill="black", font=fH)
def wrap(t, n=46):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) > n: out.append(line); line = w
        else: line = (line + " " + w).strip()
    return out + [line]
for i, s in enumerate(ORDER):
    y = HH + i * (TH + 8)
    d.text((10, y + 2), LABEL[s], fill="black", font=fB)
    if s in STY:
        for k, ln in enumerate(wrap(STY[s]["editing"], 50)[:6]): d.text((10, y + 24 + k * 13), ln, fill=(90, 90, 90), font=fS)
    for g, (tag, m, _) in enumerate(GROUPS):
        x0 = LW + g * (4 * TW + GAP)
        p = os.path.join(D, f"{m}__cafe__{tag}{s}__s1.mp4")
        if (tag and s == "none") or not os.path.exists(p):
            d.rectangle([x0, y, x0 + 4 * TW - 2, y + TH], fill=(238, 238, 238))
            d.text((x0 + 10, y + TH // 2 - 7), "control has no description" if (tag and s == "none") else "filtered by provider (no output)", fill=(100, 100, 100), font=fS); continue
        fs, fps = load(p); cs = cuts(fs)
        idx = [int(q * (len(fs) - 1)) for q in (.10, .40, .70, .95)]
        for k, j in enumerate(idx):
            im = Image.fromarray(cv2.cvtColor(fs[j], cv2.COLOR_BGR2RGB)); im.thumbnail((TW, TH))
            img.paste(im, (x0 + k * TW, y))
            if k and any(idx[k - 1] < c <= j for c in cs): d.rectangle([x0 + k * TW - 2, y, x0 + k * TW + 1, y + TH], fill=(220, 0, 0))
img.save(os.path.join(D, "sheet_cafe.jpg"), quality=88)

print(f"{'model':13s} {'cond':5s} {'style':13s} shots  cuts(s)             sat  warm  contr  sym  front  push")
for r in sorted(rows, key=lambda r: (r['model'], ORDER.index(r['style']), r['cond'])):
    print(f"{r['model'][:12]:13s} {r['cond']:5s} {r['style']:13s} {r['n_shots']:5d}  {str(r['cut_times']):18s} "
          f"{r.get('contrast',0)*0+0:4.0f}" if False else
          f"{r['model'][:12]:13s} {r['cond']:5s} {r['style']:13s} {r['n_shots']:5d}  {str(r['cut_times'])[:18]:18s} "
          f"{r.get('saturation',np.nan):5.2f} {r.get('warmth_b',np.nan):5.1f} {r.get('contrast',np.nan):5.1f} {r.get('symmetry',np.nan):5.2f} {r.get('frontal_faces',np.nan):5.2f} {r.get('push_in',np.nan):5.2f}")
