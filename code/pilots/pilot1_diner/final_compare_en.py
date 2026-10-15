"""English merged comparison: real dialogue shots (4 films, 1 s each) | name prompt | trait-table prompt.
python final_compare_en.py REFDIR GENDIR"""
import sys, os, json, glob, subprocess, cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
R, G = sys.argv[1], sys.argv[2]
ORDER = ["none", "nolan", "burton", "kubrick", "wes_anderson", "zhang_yimou", "spielberg"]
LABEL = {"none": "No style", "nolan": "Nolan", "burton": "Burton", "kubrick": "Kubrick",
         "wes_anderson": "Wes Anderson", "zhang_yimou": "Zhang Yimou", "spielberg": "Spielberg"}
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_api.json")))
TR = {k: [t.strip() for t in v.strip(", ").split(",")] for k, v in P["styles_table"].items()}
TR["none"] = ["control: no style phrase"]
MEAS = {"cool gray-blue", "low saturation", "pale characters against dark backgrounds", "high contrast", "one-point perspective",
        "centered symmetry", "slow dolly track", "frontal eye-level view", "pastel colors", "lateral tracking and pans",
        "large areas of single colors such as red and yellow", "warm backlight", "light shafts", "slow push-in to the faces", "long take with staging"}
CW, CH, LW, TH, FPS, SEC = 320, 180, 330, 76, 24, 4.0
COLS = [("real", None, "Real films: dialogue shots"), ("wan-3.0", "", "Name prompt: Wan 3.0"), ("veo-3.1-lite", "", "Name prompt: Veo 3.1 Lite"),
        ("wan-3.0", "table_", "Trait-table prompt: Wan 3.0"), ("veo-3.1-lite", "table_", "Trait-table prompt: Veo 3.1 Lite")]
GAP = [0, 14, 14, 28, 28]
A = "/System/Library/Fonts/Supplemental/Arial"
fB, fH, fS, fN = (ImageFont.truetype(A + " Bold.ttf", 20), ImageFont.truetype(A + " Bold.ttf", 15),
                  ImageFont.truetype(A + ".ttf", 13), ImageFont.truetype(A + ".ttf", 12))

def load(p):
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 24; fs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fs.append(f)
    return fs, fps

def fit(f):
    h, w = f.shape[:2]; s = min(CW / w, CH / h); im = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((CH, CW, 3), np.uint8); y, x = (CH - im.shape[0]) // 2, (CW - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im; return out

def crop_letterbox(fs):
    g = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fs[:: max(1, len(fs) // 20)]], 0)
    rows = np.where(g.mean(1) > 18)[0]
    return [f[rows[0] + 2: rows[-1] - 1] for f in fs] if len(rows) > 10 else fs

shots = pd.DataFrame(json.load(open(os.path.join(R, "dialog_shots.json"))))
idx = json.load(open(os.path.join(R, "dialog_shot_index.json")))
real = {}
for s in ORDER[1:]:
    segs, used = [], set()
    sub = shots[shots["style"] == s].sort_values("shot_seconds", ascending=False)
    for _, r in sub.iterrows():
        if r.title in used: continue
        p = [x for x in glob.glob(os.path.join(R, r.videoid + ".*")) if x.endswith((".mp4", ".webm"))][0]
        a, b, _ = max(((a, b, b - a) for _v, a, b in idx[r.videoid]), key=lambda x: x[2])     # longest dialogue shot (6 fps index)
        fs, fps = load(p); scale = fps / 6.0 if fps > 6 else 1.0
        # index was built after dropping 2 s intro at ~6 fps
        a0 = int((a / 6.0 + 2) * fps); b0 = int((b / 6.0 + 2) * fps); m = (a0 + b0) // 2
        seg = crop_letterbox(fs[max(a0, m - int(fps / 2)): min(b0, m + int(fps / 2))])
        if len(seg) < 5: continue
        segs.append(([fit(f) for f in seg], fps, r.title)); used.add(r.title)
        if len(segs) == 4: break
    real[s] = segs

def tile_text(lines):
    im = Image.new("RGB", (CW, CH), (238, 238, 238)); d = ImageDraw.Draw(im)
    for k, t in enumerate(lines): d.text((14, CH // 2 - 10 * len(lines) + k * 22), t, fill=(90, 90, 90), font=fH)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

W = LW + CW * len(COLS) + GAP[-1]; H = TH + CH * len(ORDER)
base = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(base)
d.text((12, 8), "Same content: two people talking at a diner table", fill="black", font=fH)
d.text((12, 28), "Generated: seed 1. Real: longest dialogue shot (1-3 faces)", fill=(110, 110, 110), font=fN)
d.text((12, 44), "from 4 different films, 1 s each. Traits are the literal", fill=(110, 110, 110), font=fN)
d.text((12, 58), "trait-table prompt; * = measured by a metric.", fill=(110, 110, 110), font=fN)
for j, (_, _, name) in enumerate(COLS): d.text((LW + j * CW + GAP[j] + 8, 50), name, fill="black", font=fH)
for i, s in enumerate(ORDER):
    y = TH + i * CH
    d.text((12, y + 8), LABEL[s], fill="black", font=fB)
    for k, t in enumerate(TR[s]):
        d.text((12, y + 38 + k * 19), ("* " if t in MEAS else "  ") + t, fill=(60, 60, 60) if t in MEAS else (140, 140, 140), font=fS)
base = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
GV = {(m, tag, s): (load(os.path.join(G, f"{m}__diner__{tag}{s}__s1.mp4")) if os.path.exists(os.path.join(G, f"{m}__diner__{tag}{s}__s1.mp4")) else None)
      for s in ORDER for m, tag, _ in COLS[1:]}
tmp = os.path.join(G, "_en_tmp.mp4"); vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for t in range(int(FPS * SEC)):
    fr = base.copy(); sec = t / FPS
    for i, s in enumerate(ORDER):
        y = TH + i * CH
        for j, (m, tag, _) in enumerate(COLS):
            x = LW + j * CW + GAP[j]
            if m == "real":
                if s == "none" or not real.get(s): tile = tile_text(["no real reference"] if s == "none" else ["no dialogue shot found"])
                else:
                    k = min(int(sec), len(real[s]) - 1); seg, fps_, title = real[s][k]
                    tile = seg[min(int((sec - k) * fps_), len(seg) - 1)].copy()
                    cv2.putText(tile, title[:36], (6, CH - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            elif s == "none" and tag: tile = tile_text(["control has no", "trait-table prompt"])
            else:
                v = GV[(m, tag, s)]
                tile = tile_text(["blocked by provider"]) if v is None else fit(v[0][min(int(sec * v[1]), len(v[0]) - 1)])
            fr[y:y + CH, x:x + CW] = tile
    vw.write(fr)
vw.release()
out = os.path.join(G, "final_compare_en.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out], check=True)
os.remove(tmp)
cap = cv2.VideoCapture(out); cap.set(cv2.CAP_PROP_POS_FRAMES, int(FPS * 1.5)); ok, f = cap.read()
cv2.imwrite(os.path.join(G, "final_compare_en.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90]); print("ok")
