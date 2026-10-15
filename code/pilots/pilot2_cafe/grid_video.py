"""Stitch all clips into one synchronized grid video: rows = style, cols = name/description x Wan/Veo.
python grid_video.py GENDIR"""
import sys, os, json, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
D = sys.argv[1]
ORDER = sys.argv[2].split(",") if len(sys.argv) > 2 else ["none", "nolan", "burton", "kubrick", "wes_anderson", "zhang_yimou", "spielberg"]
OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "compare_cafe_grid.mp4"
LABEL = {"none": "No style", "nolan": "Nolan", "burton": "Burton", "kubrick": "Kubrick", "wes_anderson": "Wes Anderson",
         "zhang_yimou": "Zhang Yimou", "spielberg": "Spielberg"}
COLS = [("real", None, "Real films: two-person dialogue"), ("", "wan-3.0", "Name: Wan 3.0"), ("", "veo-3.1-lite", "Name: Veo 3.1 Lite"),
        ("desc_", "wan-3.0", "Description: Wan 3.0"), ("desc_", "veo-3.1-lite", "Description: Veo 3.1 Lite")]
STY = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles_gpt54.json")))["styles"]
CW, CH, LW, TH, FPS, SEC, GAP = 448, 252, 400, 58, 24, 8.0, [0, 16, 16, 32, 32]
A = "/System/Library/Fonts/Supplemental/Arial"
fB, fH, fS = ImageFont.truetype(A + " Bold.ttf", 21), ImageFont.truetype(A + " Bold.ttf", 17), ImageFont.truetype(A + ".ttf", 13)

def load(p):
    if not os.path.exists(p): return None
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 24; fs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fs.append(cv2.resize(f, (CW, CH), interpolation=cv2.INTER_AREA))
    return fs, fps

def wrap(t, n):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n: out.append(line); line = w
        else: line = (line + " " + w).strip()
    return out + ([line] if line else [])

W = LW + CW * len(COLS) + GAP[-1]; H = TH + CH * len(ORDER)
base = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(base)
d.text((10, 6), "Talk over coffee, then stand up and leave", fill="black", font=fH)
d.text((10, 28), "Seed 1, 8 s. Description = four-system style text", fill=(110, 110, 110), font=fS)
d.text((10, 41), "written by GPT-5.4, without the director's name. Real column: 4 films, 2 s each.", fill=(110, 110, 110), font=fS)
for j, (_, _, t) in enumerate(COLS): d.text((LW + j * CW + GAP[j] + 8, 32), t, fill="black", font=fH)
for i, s in enumerate(ORDER):
    y = TH + i * CH
    d.text((10, y + 4), LABEL[s], fill="black", font=fB)
    if s in STY:
        lines = wrap(STY[s]["prompt_suffix"], 56)[:13]
        for k, ln in enumerate(lines): d.text((10, y + 32 + k * 16), ln, fill=(80, 80, 80), font=fS)
base = cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)
def tile(lines):
    im = Image.new("RGB", (CW, CH), (238, 238, 238)); dd = ImageDraw.Draw(im)
    for k, t in enumerate(lines): dd.text((14, CH // 2 - 10 + k * 20), t, fill=(90, 90, 90), font=fH)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
import glob, collections
from trait_metrics import faces
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cmd_clips/ref_dialog")
SH = json.load(open(os.path.join(R, "dialog_shots.json"))); IX = json.load(open(os.path.join(R, "dialog_shot_index.json")))
byv = collections.defaultdict(list)
for r in SH: byv[r["videoid"]].append(r)
for v in byv:
    for r, (_, a, b) in zip(byv[v], IX[v]): r["a"], r["b"] = a, b
def real_segs(style):
    cand = sorted([r for r in SH if r["style"] == style and 1.6 <= r["faces_per_frame"] <= 2.6 and r["shot_seconds"] >= 2],
                  key=lambda r: -r["shot_seconds"])
    out, used = [], set()
    for r in cand:
        if r["title"] in used: continue
        p = [x for x in glob.glob(os.path.join(R, r["videoid"] + ".*")) if x.endswith((".mp4", ".webm"))][0]
        cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 24; step = max(1, round(fps / 6)); efps = fps / step
        a0 = (r["a"] + int(2 * efps)) * step; b0 = (r["b"] + int(2 * efps)) * step; m = (a0 + b0) // 2
        if b0 > cap.get(cv2.CAP_PROP_FRAME_COUNT) - 25 * fps: continue   # MovieClips end cards
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(a0, m - int(fps))); fr = []
        for _ in range(min(b0 - max(a0, m - int(fps)), int(2 * fps))):
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        if len(fr) < 10: continue
        hits = sum(len([x for x in faces(f) if x[3] >= 0.10 * f.shape[0] and x[-1] >= 0.85]) >= 2 for f in fr[:: max(1, len(fr) // 5)][:5])
        if hits < 3: continue                                       # two clearly visible faces in most frames
        g = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr[::5]], 0); rows = np.where(g.mean(1) > 18)[0]
        if len(rows) > 10: fr = [f[rows[0] + 2: rows[-1] - 1] for f in fr]
        def fit(f):
            h, w = f.shape[:2]; k = min(CW / w, CH / h); im = cv2.resize(f, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            o = np.zeros((CH, CW, 3), np.uint8); y, x = (CH - im.shape[0]) // 2, (CW - im.shape[1]) // 2; o[y:y + im.shape[0], x:x + im.shape[1]] = im
            cv2.putText(o, r["title"][:40], (8, CH - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA); return o
        out.append(([fit(f) for f in fr], fps)); used.add(r["title"])
        if len(out) == 4: break
    return out
REAL = {s: real_segs(s) for s in ORDER if s != "none"}
V = {(tag, m, s): (None if tag == "real" else load(os.path.join(D, f"{m}__cafe__{tag}{s}__s1.mp4"))) for s in ORDER for tag, m, _ in COLS}
tmp = os.path.join(D, "_grid_tmp.mp4"); vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for t in range(int(FPS * SEC)):
    fr = base.copy()
    for i, s in enumerate(ORDER):
        for j, (tag, m, _) in enumerate(COLS):
            v = V[(tag, m, s)]
            if tag == "real":
                if s == "none": tl = tile(["no real reference"])
                else:
                    segs = REAL[s]; k = min(int(t / FPS / 2), len(segs) - 1); seg, sf = segs[k]
                    tl = seg[min(int((t / FPS - 2 * k) * sf), len(seg) - 1)]
            elif tag and s == "none": tl = tile(["control: no description"])
            elif v is None: tl = tile(["filtered by provider", "(no output)"])
            else: tl = v[0][min(int(t / FPS * v[1]), len(v[0]) - 1)]
            x = LW + j * CW + GAP[j]; fr[TH + i * CH: TH + (i + 1) * CH, x:x + CW] = tl
    vw.write(fr)
vw.release()
out = os.path.join(D, OUTNAME)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
os.remove(tmp); print(out)
