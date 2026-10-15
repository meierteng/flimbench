"""Measure real dialogue shots and generated clips with trait_metrics.
python measure_all.py REFDIR GENDIR"""
import sys, os, glob, json, cv2, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from trait_metrics import shot_metrics, faces
R, G = sys.argv[1], sys.argv[2]
SFPS = 6.0

def load(p, target=SFPS):
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 24; step = max(1, round(fps / target)); fs = []; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0: fs.append(f)
        i += 1
    return fs, fps / step

def endcard(f):
    h = cv2.cvtColor(f[:, f.shape[1] // 2:], cv2.COLOR_BGR2HSV)
    return np.mean((h[..., 1] < 25) & (h[..., 2] > 25) & (h[..., 2] < 75)) > 0.45

def shots(fs):
    def hist(f):
        g = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
        c = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(g, g).flatten(), cv2.normalize(c, c).flatten()
    H = [hist(f) for f in fs]; cut = [0]
    for i in range(1, len(H)):
        dg = cv2.compareHist(H[i - 1][0], H[i][0], cv2.HISTCMP_BHATTACHARYYA)
        dc = cv2.compareHist(H[i - 1][1], H[i][1], cv2.HISTCMP_BHATTACHARYYA)
        if max(dg, dc) > 0.33: cut.append(i)
    cut.append(len(fs)); return list(zip(cut[:-1], cut[1:]))

def real_clip(args):
    p, style, title, vid = args
    fs, fps = load(p)
    fs = fs[int(2 * fps): len(fs) - int(3 * fps)]
    ec = [i for i in range(max(0, len(fs) - int(60 * fps)), len(fs)) if endcard(fs[i])]
    if ec: fs = fs[: ec[0]]
    if len(fs) < 10: return [], []
    g = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fs[:: max(1, len(fs) // 40)]], 0)
    rows = np.where(g.mean(1) > 18)[0]; cols = np.where(g.mean(0) > 18)[0]
    t, b = (rows[0] + 2, rows[-1] - 1) if len(rows) > 10 else (0, g.shape[0])
    l, r = (cols[0] + 2, cols[-1] - 1) if len(cols) > 10 else (0, g.shape[1])
    aspect = (r - l) / max(1, b - t)
    fs = [f[t:b, l:r] for f in fs]
    out, thumbs = [], []
    for a, b2 in shots(fs):
        seg = fs[a:b2]
        if len(seg) < SFPS * 1.0: continue
        probe = seg[:: max(1, len(seg) // 5)]
        nf = [len(faces(f)) for f in probe]
        if np.mean([1 <= n <= 3 for n in nf]) < 0.6: continue          # dialogue shot: 1-3 faces in most frames
        m = shot_metrics(seg, fps)
        if m["lateral"] > 0.9 and m["push_in"] != m["push_in"]: continue
        out.append(dict(source="real", style=style, title=title, videoid=vid, aspect=round(aspect, 2), **m))
        thumbs.append((vid, a, b2))
    return out, thumbs

def gen_clip(p):
    base = os.path.basename(p)[:-4]; m, c, s, sd = base.split("__")
    kind = "table" if s.startswith("table_") else "traits_old" if s.startswith("traits_") else "name"
    style = s.split("_", 1)[1] if kind != "name" else s
    if kind == "name" and s == "none": kind, style = "none", "none"
    fs, fps = load(p)
    return dict(source=m, prompt=kind, style=style, seed=sd, **shot_metrics(fs, fps))

if __name__ == "__main__":
    pk = pd.read_csv(os.path.join(R, "picked.csv"))
    jobs = []
    for _, r in pk.iterrows():
        p = [x for x in glob.glob(os.path.join(R, r.videoid + ".*")) if x.endswith((".mp4", ".webm"))]
        if p: jobs.append((p[0], r["style"], r.title, r.videoid))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(real_clip, jobs))
    real = [x for o, _ in res for x in o]
    thumbs = {j[3]: t for j, (_, t) in zip(jobs, res)}
    json.dump(real, open(os.path.join(R, "dialog_shots.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(thumbs, open(os.path.join(R, "dialog_shot_index.json"), "w"))
    gp = sorted(glob.glob(os.path.join(G, "*__diner__*.mp4")))
    with ProcessPoolExecutor(6) as ex:
        gen = list(ex.map(gen_clip, gp))
    json.dump(gen, open(os.path.join(G, "gen_trait_metrics.json"), "w"), indent=1)
    df = pd.DataFrame(real)
    print(df.groupby("style").agg(shots=("title", "size"), clips=("videoid", "nunique"), films=("title", "nunique")))
