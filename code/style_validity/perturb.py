"""Known-perturbation (direction) test for extract.shot_metrics.
Each edit is applied at strengths 0..3 to every analysis frame (and native frame) of real shots; the target metric must move
monotonically in the expected direction. Also reports selectivity (which other metrics move) and a grain-survives-compression test.

python perturb.py CLIPLIST CLIPDIR OUTDIR --clips 16 --workers 8"""
import os, sys, json, argparse, subprocess, random, cv2, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as X

LEVELS = [0, 1, 2, 3]
Yw = np.array([0.0722, 0.7152, 0.2126])         # BGR luma weights


def luma(x): return (x @ Yw)[..., None]


# ---- per-frame edits: fn(frame_float01, k, t, n, masks) ; t = frame index in shot, n = shot length
def warm(x, k, *a): return x * np.array([1 - .04 * k, 1, 1 + .04 * k])
def split(x, k, *a):
    w = luma(x); s = .05 * k
    return x + s * (w * np.array([-1, -.2, 1]) + (1 - w) * np.array([1, .3, -1]) * .6)
def side_light(x, k, *a):                       # brighter and warmer light from camera left (position dependent)
    W = x.shape[1]; ramp = np.linspace(1, -1, W)[None, :, None]
    return x * (1 + .12 * k * ramp) * (1 + .04 * k * np.clip(ramp, 0, 1) * np.array([-1, 0, 1]))
def contrast(x, k, *a):                        # linear contrast around the frame mean luminance
    c = float(luma(x).mean()); return c + (x - c) * (1 + .25 * k)
def saturate(x, k, *a): return luma(x) + (x - luma(x)) * (1 + .25 * k)
def lift(x, k, *a): s = .03 * k; return s + (1 - s) * x
def vignette(x, k, *a):
    h, w = x.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]; r2 = (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    return x * (1 - .15 * k * r2)[..., None]
def zoom(x, k, *a):
    h, w = x.shape[:2]; c = 1 / (1 + .15 * k); hh, ww = int(h * c), int(w * c); y0, x0 = (h - hh) // 2, (w - ww) // 2
    return cv2.resize(x[y0:y0 + hh, x0:x0 + ww], (w, h))
def roll(x, k, *a):
    h, w = x.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5 * k, 1.15)
    return cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REFLECT)
def blur_bg(x, k, t, n, m):                    # shallower depth of field: blur outside the person region
    if k == 0 or m is None: return x
    sig = 1.2 * k * x.shape[1] / X.WORK_W; b = cv2.GaussianBlur(x, (0, 0), sig)
    mm = cv2.resize(m.astype(np.float32), (x.shape[1], x.shape[0]))[..., None]
    return x * mm + b * (1 - mm)
def blur_all(x, k, *a):
    if k == 0: return x
    return cv2.GaussianBlur(x, (0, 0), .6 * k * x.shape[1] / X.WORK_W)
def grain(x, k, t, *a):                         # luminance grain, strongest in midtones like film
    rng = np.random.default_rng(1000 * t + 7); g = rng.normal(0, 1, x.shape[:2]).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), .7 * x.shape[1] / 1920 + .3)
    w = 4 * luma(x) * (1 - luma(x)); return x + (.012 * k) * g[..., None] * w * 3
def halation(x, k, *a):
    hl = (luma(x)[..., 0] > .92).astype(np.float32); glow = cv2.GaussianBlur(hl, (0, 0), 6 * x.shape[1] / 1920 + 1)
    return x + (.08 * k) * glow[..., None] * np.array([0, .25, 1])
def mirror(x, k, *a):                           # more symmetric: blend with the mirror image
    return x * (1 - .15 * k) + x[:, ::-1] * .15 * k
def pan(x, k, t, n, *a):                        # synthetic pan: crop window slides right at 0.04 frame widths per second per level
    h, w = x.shape[:2]; c = .6; ww, hh = int(w * c), int(h * c); x0 = int(min(.04 * k * t / X.SFPS, 1 - c) * w); y0 = (h - hh) // 2
    return cv2.resize(x[y0:y0 + hh, x0:x0 + ww], (w, h))
def push(x, k, t, n, *a):                       # synthetic push-in: zoom grows over the shot
    return zoom(x, k * t / max(1, n - 1))
def coarse_grain(x, k, t, *a):                 # same grain strength, growing grain size
    rng = np.random.default_rng(1000 * t + 7); g = rng.normal(0, 1, x.shape[:2]).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), (.5 + .7 * k) * x.shape[1] / 1280); g /= g.std() + 1e-6
    return x + .025 * g[..., None]
def shake(x, k, t, *a):
    rng = np.random.default_rng(t + 11); h, w = x.shape[:2]; d = rng.normal(0, .006 * k, 2) * w
    M = np.float32([[1.06, 0, -.03 * w + d[0]], [0, 1.06, -.03 * h + d[1]]]); return cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REFLECT)

EDITS = {   # name: (fn, target metric, expected sign, apply to native frames too)
    "warm": (warm, "CL4_warmth_b", +1), "split_tone": (split, "GR7_split_b", +1), "side_light": (side_light, "LI3_key_side", +1),
    "side_light_spread": (side_light, "LI8_light_color_spread", +1), "contrast": (contrast, "GR2_contrast", +1),
    "saturation": (saturate, "CL5_chroma", +1), "colorfulness": (saturate, "CL6_colorfulness", +1),
    "lift_black": (lift, "GR3_black_p2", +1), "vignette": (vignette, "GR13_vignette", -1), "zoom": (zoom, "CA1_face_size", +1),
    "roll": (roll, "CA4_roll_deg", +1), "blur_bg": (blur_bg, "CA8_dof_ratio", +1), "blur_all": (blur_all, "TX5_edge_width", +1), "grain_size": (coarse_grain, "TX2_grain_size", -1),
    "grain": (grain, "TX1_grain", +1), "halation": (halation, "TX6_halation", +1), "mirror": (mirror, "CO2_symmetry", +1),
    "pan": (pan, "CM5_pan_signed", -1), "push": (push, "CM6_zoom_scale", +1), "shake": (shake, "CM4_shake", +1),
}
WATCH = ["CL4_warmth_b", "CL4_tint_a", "CL5_chroma", "CL6_colorfulness", "GR1_luma", "GR2_contrast", "GR3_black_p2", "GR4_white_p98",
         "GR7_split_b", "GR12_lum_explained_b", "GR13_vignette", "LI3_key_side", "LI2_face_lr_stops", "LI8_light_color_spread",
         "CA1_face_size", "CA4_roll_deg", "CA8_dof_ratio", "CO2_symmetry", "TX1_grain", "TX2_grain_size", "TX5_edge_width", "TX6_halation", "CM5_pan_signed",
         "CM3_cam_speed", "CM5_pan", "CM6_zoom_scale", "CM4_shake"]


def pick_shots(path, n_shots=3):
    small, native, fps, thumbs, step = X.read_clip(path)
    s0, s1 = int(2 * fps), len(small) - int(3 * fps)
    fs = small[s0:s1]; t, b, l, r = X.crop_box(fs); fs = [f[t:b, l:r] for f in fs]
    nat = {}
    for k, f in native.items():
        if s0 <= k < s1:
            sc = f.shape[1] / X.WORK_W; nat[k - s0] = f[int(t * sc): int(b * sc), int(l * sc): int(r * sc)]
    scenes, _ = X.transnet_cuts(thumbs); out = []
    for st, en in scenes:
        a, b2 = max(int(np.ceil(st / step)) - s0, 0), min(en // step + 1 - s0, len(fs))
        if b2 - a < 2 * X.SFPS: continue
        seg = fs[a:b2]
        if len(X.faces(seg[len(seg) // 2])) == 0: continue          # need a face for face-based targets
        out.append((seg, [nat[k] for k in range(a, b2) if k in nat], fps))
    random.Random(0).shuffle(out); return out[:n_shots]


def run_clip(args):
    path, vid = args; rows = []
    for si, (seg, nseg, fps) in enumerate(pick_shots(path)):
        d0 = X.faces(seg[len(seg) // 2]); m = X.body_mask(seg[0].shape, d0)
        base = [f.astype(np.float32) / 255 for f in seg]; nbase = [f.astype(np.float32) / 255 for f in nseg]
        for name, (fn, tgt, sign) in EDITS.items():
            for k in LEVELS:
                ed = [(np.clip(fn(x, k, t, len(base), m), 0, 1) * 255).astype(np.uint8) for t, x in enumerate(base)]
                ned = [(np.clip(fn(x, k, min(t * X.TEX_EVERY, len(base) - 1), len(base), m), 0, 1) * 255).astype(np.uint8) for t, x in enumerate(nbase)]
                met, _ = X.shot_metrics(ed, ned, fps)
                rows.append(dict(shot=f"{vid}_{si}", edit=name, level=k, **{w: met.get(w, np.nan) for w in WATCH}))
        rows += grain_compression(nseg, f"{vid}_{si}")
    return rows


def grain_compression(nseg, shot_id):
    """Add grain at known strength, encode with x264 at a YouTube-like bitrate, decode, and re-measure TX1."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe(); rows = []
    if len(nseg) < 2: return rows
    frames = [f.astype(np.float32) / 255 for f in nseg[:4]]
    h, w = frames[0].shape[:2]; h -= h % 2; w -= w % 2
    for k in LEVELS:
        seq = []
        for t, x in enumerate(frames):
            g = (np.clip(grain(x, k, t), 0, 1) * 255).astype(np.uint8)[:h, :w]
            seq += [g] * 6                                         # 6 identical frames per still keeps inter-frame prediction realistic
        kbps = int(1500 * (w * h) / (1920 * 1080)) + 300
        p = subprocess.run([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                            "-c:v", "libx264", "-b:v", f"{kbps}k", "-preset", "medium", "-f", "matroska", "-"],
                           input=b"".join(s.tobytes() for s in seq), capture_output=True)
        d = subprocess.run([ff, "-loglevel", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], input=p.stdout, capture_output=True)
        dec = np.frombuffer(d.stdout, np.uint8).reshape(-1, h, w, 3)[3::6]
        tx = X.texture(list(dec))
        rows.append(dict(shot=shot_id, edit="grain_after_x264", level=k, TX1_grain=tx.get("TX1_grain", np.nan)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cliplist"); ap.add_argument("clipdir"); ap.add_argument("outdir")
    ap.add_argument("--clips", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    cl = pd.read_csv(a.cliplist)
    files = {os.path.splitext(x)[0]: os.path.join(a.clipdir, x) for x in os.listdir(a.clipdir) if x.endswith((".mp4", ".webm"))}
    cl = cl[cl.videoid.isin(files)].sample(frac=1, random_state=0).groupby("style").head(max(1, a.clips // 6))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(run_clip, [(files[v], v) for v in cl.videoid]) for r in rs]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "perturb_raw.csv"), index=False)
    allE = dict(EDITS); allE["grain_after_x264"] = (None, "TX1_grain", +1)
    sens = []
    for name, (_, tgt, sign) in allE.items():
        d = df[df.edit == name].pivot(index="shot", columns="level", values=tgt).dropna()
        if d.empty: sens.append(dict(edit=name, target=tgt, n=0)); continue
        diffs = np.diff(d.values, axis=1) * sign
        sd0 = df[(df.edit == "warm") & (df.level == 0)][tgt].std()
        sens.append(dict(edit=name, target=tgt, n=len(d), monotonic=float(np.mean((diffs > 0).all(1))),
                         right_direction_l3=float(np.mean((d[3] - d[0]) * sign > 0)), effect_l3_in_sd=float(((d[3] - d[0]) * sign).mean() / (sd0 + 1e-9))))
    sens = pd.DataFrame(sens)
    base = df[(df.edit == "warm") & (df.level == 0)].set_index("shot")[WATCH]; sd = base.std()
    sel = {}
    for name in EDITS:
        a3 = df[(df.edit == name) & (df.level == 3)].set_index("shot")[WATCH]; a0 = df[(df.edit == name) & (df.level == 0)].set_index("shot")[WATCH]
        sel[name] = ((a3 - a0).mean() / sd).round(2)
    sel = pd.DataFrame(sel).T
    gl = df[df.edit.isin(["split_tone", "side_light"]) & df.level.isin([0, 3])].pivot_table(index=["edit", "shot"], columns="level", values="GR12_lum_explained_b")
    gl["delta"] = gl[3] - gl[0]
    gvl = gl.groupby("edit")["delta"].describe()[["count", "mean", "std"]]
    sens.to_csv(os.path.join(a.outdir, "sensitivity.csv"), index=False); sel.to_csv(os.path.join(a.outdir, "selectivity.csv")); gvl.to_csv(os.path.join(a.outdir, "grade_vs_light.csv"))
    pd.set_option("display.width", 250)
    print(sens.round(3).to_string(index=False)); print(); print(sel.to_string()); print(); print(gvl.round(3).to_string())
