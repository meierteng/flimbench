"""Phase 1 (CPU) extraction of the metric pool in note 21 for CMD clips.
Every shot of at least 1 s is measured (not only dialogue shots); `dialog` flags shots with 1 to 3 faces in most frames.
Metric ids follow note 21. Model-based metrics (TransNetV2, FilmOps, ShotVL, ACaM, GeoCalib, Depth Pro) are phase 2.

python extract.py CLIPLIST CLIPDIR OUTDIR [--n N] [--workers W]
Writes OUTDIR/shots/<videoid>.json (one file per clip, so a crashed job resumes) and OUTDIR/clip_errors.txt."""
import os, sys, json, argparse, traceback, cv2, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
SFPS = 6.0          # analysis frame rate
WORK_W = 480        # working width for most metrics
TEX_EVERY = 6       # keep a native-resolution frame every 6 analysis frames (1 per second) for texture metrics
_YN = None


# ---------------------------------------------------------------- io
def _frames(path):
    """Yield BGR frames decoded by ffmpeg with an explicit YUV to RGB matrix (BT.709 for HD, BT.601 below 720 lines) and
    limited input range. OpenCV ignores the color tags and decodes HD H.264 with BT.601, which shifts hues (note 25a)."""
    import subprocess, imageio_ffmpeg
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    mat = "bt709" if h >= 720 else "bt601"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-threads", "1", "-i", path, "-an",
           "-vf", f"scale=in_color_matrix={mat}:in_range=tv:out_range=pc,format=bgr24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    def gen():
        n = w * h * 3
        try:
            while True:
                b = p.stdout.read(n)
                if len(b) < n: return
                yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
        finally:
            p.stdout.close(); p.wait()
    return gen(), fps


def read_clip(path):
    it, fps = _frames(path); step = max(1, round(fps / SFPS))
    small, native, thumbs, i, k = [], {}, [], 0, 0
    for f in it:
        thumbs.append(cv2.resize(f, (48, 27), interpolation=cv2.INTER_AREA)[..., ::-1])
        if i % step == 0:
            small.append(cv2.resize(f, (WORK_W, int(round(WORK_W * f.shape[0] / f.shape[1]))), interpolation=cv2.INTER_AREA))
            if k % TEX_EVERY == 0: native[k] = f.copy()
            k += 1
        i += 1
    return small, native, fps / step, np.stack(thumbs) if thumbs else None, step


_TN = None
def transnet_cuts(thumbs):
    """TransNetV2 (Soucek and Lokoc 2020) on native-rate 48x27 RGB frames. Returns scenes [start, end] in native frames
    and the number of gradual transitions (boundary runs of at least 3 frames above threshold)."""
    global _TN
    import torch
    torch.set_num_threads(1)
    if _TN is None:
        from transnetv2_pytorch import TransNetV2
        _TN = TransNetV2(device="cpu"); _TN.eval()
    single, many = _TN.predict_frames(torch.from_numpy(np.ascontiguousarray(thumbs)), quiet=True)
    p = single.numpy(); scenes = _TN.predictions_to_scenes(p, 0.5)
    above = (p > 0.5).astype(int); runs = np.diff(np.r_[0, above, 0]); starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
    gradual = int(np.sum((ends - starts) >= 3))
    return scenes, gradual


def endcard(f):
    h = cv2.cvtColor(f[:, f.shape[1] // 2:], cv2.COLOR_BGR2HSV)
    return np.mean((h[..., 1] < 25) & (h[..., 2] > 25) & (h[..., 2] < 75)) > 0.45


def crop_box(fs):
    g = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fs[:: max(1, len(fs) // 40)]], 0)
    rows = np.where(g.mean(1) > 18)[0]; cols = np.where(g.mean(0) > 18)[0]
    if len(rows) < 10 or len(cols) < 10: return 0, g.shape[0], 0, g.shape[1]
    return rows[0] + 2, rows[-1] - 1, cols[0] + 2, cols[-1] - 1


def shot_bounds(fs):
    def hist(f):
        g = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
        c = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(g, g).flatten(), cv2.normalize(c, c).flatten()
    H = [hist(f) for f in fs]; cut = [0]
    for i in range(1, len(H)):
        d = max(cv2.compareHist(H[i - 1][0], H[i][0], cv2.HISTCMP_BHATTACHARYYA),
                cv2.compareHist(H[i - 1][1], H[i][1], cv2.HISTCMP_BHATTACHARYYA))
        if d > 0.33: cut.append(i)
    cut.append(len(fs)); return list(zip(cut[:-1], cut[1:]))


# ---------------------------------------------------------------- helpers
def faces(f):
    global _YN
    if _YN is None:
        _YN = cv2.FaceDetectorYN.create(os.path.join(HERE, "yunet.onnx"), "", (320, 320), 0.7)
    _YN.setInputSize((f.shape[1], f.shape[0])); _, d = _YN.detect(f)
    return [] if d is None else [x for x in d if x[3] >= 0.08 * f.shape[0]]


def yaw_proxy(d):
    re_, le, nose = d[4:6], d[6:8], d[8:10]
    return (nose[0] - (re_[0] + le[0]) / 2) / (np.linalg.norm(le - re_) + 1e-6)


def lab_f(f):          # float CIELAB: L 0..100, a and b roughly -100..100
    return cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def body_mask(shape, dets):
    """Person region proxy: face box widened to 3 face widths and extended to the bottom of the frame."""
    m = np.zeros(shape[:2], bool)
    for x in dets:
        x0, y0, w0, h0 = x[:4]; cx = x0 + w0 / 2
        m[max(0, int(y0 - .3 * h0)):, max(0, int(cx - 1.5 * w0)): int(cx + 1.5 * w0)] = True
    return m


def face_mask(shape, dets):
    m = np.zeros(shape[:2], bool)
    for x in dets:
        x0, y0, w0, h0 = [int(v) for v in x[:4]]; m[max(0, y0): y0 + h0, max(0, x0): x0 + w0] = True
    return m


def circ_dist(a, b): return np.abs((a - b + 180) % 360 - 180)


def spectral_saliency(gray):
    g = cv2.resize(gray, (64, 64)).astype(np.float32)
    F = np.fft.fft2(g); A = np.log(np.abs(F) + 1e-6); P = np.angle(F)
    S = np.abs(np.fft.ifft2(np.exp(A - cv2.blur(A, (3, 3)) + 1j * P))) ** 2
    S = cv2.GaussianBlur(S.astype(np.float32), (9, 9), 2.5)
    return cv2.resize(S / (S.max() + 1e-9), (gray.shape[1], gray.shape[0]))


HARMONY = {"i": [(0, 18)], "V": [(0, 93.6)], "L": [(0, 18), (90, 79.2)], "I": [(0, 18), (180, 18)],
           "T": [(0, 180)], "Y": [(0, 93.6), (180, 18)], "X": [(0, 93.6), (180, 93.6)]}


def harmony_fit(hue, w):
    """Cohen-Or et al. 2006 hue templates: chroma-weighted mean distance (deg) of hues to the best rotated template."""
    hist = np.bincount(np.clip(hue.astype(int) % 360, 0, 359), w, 360); hs = np.arange(360) + .5
    best = (1e9, "")
    for name, secs in HARMONY.items():
        for rot in range(0, 360, 6):
            d = np.full(360, 1e9)
            for off, width in secs:
                d = np.minimum(d, np.maximum(circ_dist(hs, rot + off) - width / 2, 0))
            s = float((hist * d).sum() / (hist.sum() + 1e-9))
            if s < best[0]: best = (s, name)
    return best


# ---------------------------------------------------------------- shot metrics
def shot_metrics(fs, natives, fps):
    out = {}; n = len(fs); h, w = fs[0].shape[:2]
    probe_idx = list(range(0, n, max(1, n // 8)))
    dets = {i: faces(fs[i]) for i in probe_idx}
    nf = [len(dets[i]) for i in probe_idx]
    out["dialog"] = bool(np.mean([1 <= x <= 3 for x in nf]) >= 0.6)
    labs = {i: lab_f(fs[i]) for i in probe_idx}

    # ---- staging and composition from faces (ST, CO, CA1)
    sizes, xs, ys, yaws, tops, look, dist, ratio, area = [], [], [], [], [], [], [], [], []
    for i in probe_idx:
        d = dets[i]
        if not d: continue
        big = max(d, key=lambda x: x[3])
        sizes.append(big[3] / h); cx = (big[0] + big[2] / 2) / w; xs.append(cx); ys.append((big[1] + big[3] / 2) / h)
        tops.append(big[1] / h); yp = yaw_proxy(big); yaws.extend(abs(yaw_proxy(x)) for x in d)
        if abs(yp) > .08: look.append((1 - cx) if yp > 0 else cx)      # space in front of the face
        area.append(body_mask(fs[i].shape, d).mean())
        if len(d) >= 2:
            a, b = sorted(d, key=lambda x: -x[3])[:2]
            ca, cb = np.array([a[0] + a[2] / 2, a[1] + a[3] / 2]), np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
            dist.append(np.linalg.norm(ca - cb) / ((a[3] + b[3]) / 2)); ratio.append(a[3] / b[3])
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    out.update(ST1_faces=float(np.mean(nf)), CA1_face_size=med(sizes), ST2_face_x=med(xs), ST2_face_y=med(ys),
               ST3_yaw_abs=med(yaws), ST4_frontal=float(np.mean(np.array(yaws) < .15)) if yaws else np.nan,
               ST3_profile=float(np.mean(np.array(yaws) > .45)) if yaws else np.nan,
               ST5_pair_dist=med(dist), ST6_size_ratio=med(ratio), ST7_person_area=med(area),
               CO4_thirds=med([min(abs(x - 1 / 3), abs(x - 2 / 3)) for x in xs]), CO4_center=med([abs(x - .5) for x in xs]),
               CO5_headroom=med(tops), CO6_lookroom=med(look))
    out["CA2_shot_type"] = ("none" if np.median(nf) == 0 else "single" if np.median(nf) == 1 else "two" if np.median(nf) == 2 else "group")

    # ---- tone, color, grading on probe frames (CL, GR, LI)
    L = np.concatenate([labs[i][..., 0].ravel()[::3] for i in probe_idx])
    A = np.concatenate([labs[i][..., 1].ravel()[::3] for i in probe_idx])
    B = np.concatenate([labs[i][..., 2].ravel()[::3] for i in probe_idx])
    C = np.hypot(A, B); Hh = np.degrees(np.arctan2(B, A)) % 360
    out["black"] = bool(L.mean() < 4)
    out.update(GR1_luma=float(L.mean()), GR1_median=float(np.median(L)), GR2_contrast=float(np.percentile(L, 95) - np.percentile(L, 5)),
               GR3_black_p2=float(np.percentile(L, 2)), GR3_crush=float(np.mean(L < 3)), GR4_white_p98=float(np.percentile(L, 98)),
               GR4_clip=float(np.mean(L > 97)), GR6_skew=float(((L - L.mean()) ** 3).mean() / (L.std() ** 3 + 1e-6)),
               LI1_dark_share=float(np.mean(L < 20)), LI1_bright_share=float(np.mean(L > 80)))
    hd, _ = np.histogram(L, bins=27, range=(70, 97)); ok_ = hd > 0          # log-density slope of the top of the histogram
    out["GR5_rolloff_slope"] = float(np.polyfit(np.arange(27)[ok_], np.log(hd[ok_]), 1)[0]) if ok_.sum() >= 8 and (L >= 70).sum() > 300 else np.nan
    out.update(CL4_warmth_b=float(B.mean()), CL4_tint_a=float(A.mean()), CL5_chroma=float(C.mean()))
    hsv = np.concatenate([cv2.cvtColor(fs[i], cv2.COLOR_BGR2HSV)[..., 1].ravel()[::3] for i in probe_idx])
    out["CL5_saturation"] = float(hsv.mean() / 255)
    rgs, ybs = [], []
    for i in probe_idx:
        f = fs[i].astype(np.float32); rg = f[..., 2] - f[..., 1]; yb = .5 * (f[..., 2] + f[..., 1]) - f[..., 0]
        rgs.append(rg.ravel()[::3]); ybs.append(yb.ravel()[::3])
    rg, yb = np.concatenate(rgs), np.concatenate(ybs)
    out["CL6_colorfulness"] = float(np.hypot(rg.std(), yb.std()) + .3 * np.hypot(rg.mean(), yb.mean()))
    ch = C > 10
    hh = np.bincount((Hh[ch] / 10).astype(int) % 36, minlength=36).astype(float); p = hh / max(hh.sum(), 1)
    out["CL7_hue_entropy"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum()) if ch.sum() > 500 else np.nan
    warm = ch & ((Hh < 100) | (Hh > 330)); cool = ch & (Hh > 160) & (Hh < 290)
    out["CL8_warm_cool"] = float(np.log((warm.sum() + 1) / (cool.sum() + 1)))
    if ch.sum() > 500:                                       # axial hue concentration: two opposite hues dominate (e.g. orange and teal)
        z = (C[ch] * np.exp(2j * np.radians(Hh[ch]))).sum() / C[ch].sum()
        out["CL9_axial_conc"] = float(np.abs(z)); out["CL9_axis_deg"] = float(np.degrees(np.angle(z)) / 2 % 180)
    else:
        out["CL9_axial_conc"] = out["CL9_axis_deg"] = np.nan
    if ch.sum() > 500:
        s, t = harmony_fit(Hh[ch], C[ch]); out["CL10_harmony_resid"] = s; out["CL10_harmony_type"] = t
    else:
        out["CL10_harmony_resid"] = np.nan; out["CL10_harmony_type"] = "achromatic"
    out["CL12_pastel"] = float(np.mean((L > 65) & (C > 8) & (C < 30)))
    out["CL12_red_yellow"] = float(np.mean((C > 40) & ((Hh < 30) | (Hh > 340) | ((Hh > 60) & (Hh < 100)))))
    # palette (k-means, 5 colors) on one middle frame
    mid = labs[probe_idx[len(probe_idx) // 2]].reshape(-1, 3)[::7].astype(np.float32)
    _, lab_k, cen = cv2.kmeans(mid, 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .5), 2, cv2.KMEANS_PP_CENTERS)
    wts = np.bincount(lab_k.ravel(), minlength=5) / len(lab_k); o = np.argsort(-wts)
    out["CL1_palette"] = [[round(float(v), 1) for v in cen[j]] + [round(float(wts[j]), 3)] for j in o]
    dom = cen[o[0]]; out.update(CL1_dom_L=float(dom[0]), CL1_dom_a=float(dom[1]), CL1_dom_b=float(dom[2]))
    # person vs background (CL2, CL3, CL11, LI9) and skin (GR9)
    pl, bl, skin, fl, bgl, lr, tb, rim, grad_ratio = [], [], [], [], [], [], [], [], []
    for i in probe_idx:
        d = dets[i]
        if not d: continue
        lab = labs[i]; bm = body_mask(lab.shape, d); fm = face_mask(lab.shape, d)
        if bm.mean() < .95: pl.append(lab[bm].mean(0)); bl.append(lab[~bm].mean(0))
        fl.append(lab[fm][:, 0].mean()); bgl.append(np.median(lab[~fm][:, 0]))
        big = max(d, key=lambda x: x[3]); x0, y0, w0, h0 = [int(v) for v in big[:4]]
        x0, y0 = max(0, x0), max(0, y0); face = lab[y0: y0 + h0, x0: x0 + w0]
        if face.size == 0: continue
        core = face[h0 // 4: 3 * h0 // 4, w0 // 4: 3 * w0 // 4]
        if core.size:
            cm = core.reshape(-1, 3).mean(0)
            if cm[0] > 20 and np.hypot(cm[1], cm[2]) > 5: skin.append(cm)   # skip faces too dark or too gray to read hue
        Lf = face[..., 0]; Y = ((Lf + 16) / 116) ** 3                       # relative luminance from L*
        lh, rh = Y[:, : Lf.shape[1] // 2].mean(), Y[:, Lf.shape[1] // 2:].mean()
        lr.append(np.log2((lh + 1e-4) / (rh + 1e-4)))                        # >0: camera-left side brighter
        tb.append(np.log2((Y[: Lf.shape[0] // 2].mean() + 1e-4) / (Y[Lf.shape[0] // 2:].mean() + 1e-4)))
        gy, gx = np.gradient(Lf); gm = np.hypot(gx, gy).mean(); grad_ratio.append(gm / (Lf.std() + 1e-3))
        ring = np.zeros(lab.shape[:2], bool); pad = int(.25 * h0)
        ring[max(0, y0 - pad): y0 + h0, max(0, x0 - pad): x0 + w0 + pad] = True; ring &= ~fm
        if ring.sum() > 50: rim.append(np.percentile(lab[ring][:, 0], 95) - Lf.mean())
    if pl:
        pm, bm_ = np.mean(pl, 0), np.mean(bl, 0)
        out.update(CL2_person_L=float(pm[0]), CL2_person_a=float(pm[1]), CL2_person_b=float(pm[2]),
                   CL3_bg_L=float(bm_[0]), CL3_bg_a=float(bm_[1]), CL3_bg_b=float(bm_[2]),
                   CL11_fig_ground_dE=float(np.linalg.norm(pm - bm_)),
                   CL9_person_bg_hue=float(circ_dist(np.degrees(np.arctan2(pm[2], pm[1])), np.degrees(np.arctan2(bm_[2], bm_[1])))))
    out["LI9_face_minus_bg"] = float(np.mean(np.array(fl) - np.array(bgl))) if fl else np.nan
    if skin:
        s = np.mean(skin, 0); out.update(GR9_skin_hue=float(np.degrees(np.arctan2(s[2], s[1]))), GR9_skin_chroma=float(np.hypot(s[1], s[2])))
    out.update(LI2_face_lr_stops=float(np.median(np.abs(lr))) if lr else np.nan, LI3_key_side=float(np.median(lr)) if lr else np.nan,
               LI3_top_bottom=float(np.median(tb)) if tb else np.nan, LI4_rim=float(np.median(rim)) if rim else np.nan,
               LI5_shadow_sharpness=float(np.median(grad_ratio)) if grad_ratio else np.nan)
    # neutral-pixel cast by luminance band (GR7, GR8), luminance-explained share (GR12), highlight desaturation (GR11)
    # cast per luminance band = mode of the (a, b) histogram in that band, no chroma threshold (note 25a: a C*<12
    # neutral filter drops exactly the pixels a strong grade has tinted)
    for name, lo, hi_ in [("sh", 8, 35), ("mid", 35, 65), ("hi", 65, 95)]:
        m = (L >= lo) & (L < hi_)
        if m.sum() > 300:
            H2, ae, be = np.histogram2d(A[m], B[m], bins=60, range=[[-30, 30], [-30, 30]])
            H2 = cv2.GaussianBlur(H2.astype(np.float32), (0, 0), 1.5); i_, j_ = np.unravel_index(H2.argmax(), H2.shape)
            ma, mb = (ae[i_] + ae[i_ + 1]) / 2, (be[j_] + be[j_ + 1]) / 2
            near = m & (np.hypot(A - ma, B - mb) < 4)                     # refine the mode with a local mean
            out[f"GR8_neutral_a_{name}"] = float(A[near].mean()); out[f"GR8_neutral_b_{name}"] = float(B[near].mean())
        else:
            out[f"GR8_neutral_a_{name}"] = out[f"GR8_neutral_b_{name}"] = np.nan
    out["GR7_split_b"] = out["GR8_neutral_b_hi"] - out["GR8_neutral_b_sh"]
    out["GR7_split_a"] = out["GR8_neutral_a_hi"] - out["GR8_neutral_a_sh"]
    if skin and np.isfinite(out["GR8_neutral_b_mid"]):
        out["GR10_skin_minus_neutral_b"] = float(s[2] - out["GR8_neutral_b_mid"]); out["GR10_skin_minus_neutral_a"] = float(s[1] - out["GR8_neutral_a_mid"])
    bins = np.clip((L / 10).astype(int), 0, 9); cnt = np.maximum(np.bincount(bins, None, 10), 1)
    for nm, v in [("b", B), ("a", A)]:
        mu = np.bincount(bins, v, 10) / cnt; out[f"GR12_lum_explained_{nm}"] = float(1 - np.var(v - mu[bins]) / (np.var(v) + 1e-6))
    md, hm = (L > 30) & (L < 70), L >= 70
    out["GR11_hi_desat"] = float(C[hm].mean() / (C[md].mean() + 1e-6)) if hm.sum() > 300 and md.sum() > 300 else np.nan
    vg = []
    for i in probe_idx:
        Lm = labs[i][..., 0]; ch_, cw = h // 6, w // 6
        vg.append(np.mean([Lm[:ch_, :cw].mean(), Lm[:ch_, -cw:].mean(), Lm[-ch_:, :cw].mean(), Lm[-ch_:, -cw:].mean()]) -
                  Lm[h // 3: 2 * h // 3, w // 3: 2 * w // 3].mean())          # corners minus center, L* units
    out["GR13_vignette"] = float(np.mean(vg))
    # light color spatial variance (LI8): grey-edge illuminant per 3x3 block, spread of chromaticity
    ill = []
    for i in probe_idx[:4]:
        f = cv2.GaussianBlur(fs[i].astype(np.float32), (0, 0), 1)
        gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0)); gy = np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1)); g = gx + gy
        for by in range(3):
            for bx in range(3):
                e = g[by * h // 3: (by + 1) * h // 3, bx * w // 3: (bx + 1) * w // 3].reshape(-1, 3).mean(0) + 1e-3
                ill.append(np.log(e[2] / e[1])), ill.append(np.log(e[0] / e[1]))
    ill = np.array(ill).reshape(-1, 2); out["LI8_light_color_spread"] = float(ill.std(0).mean())
    out["LI7_practicals"] = float(np.mean([len([c for c in cv2.connectedComponentsWithStats(
        (labs[i][..., 0] > 95).astype(np.uint8))[2][1:, 4] if 3 < c < .005 * h * w]) for i in probe_idx]))
    cs = [labs[i][..., 1:].reshape(-1, 2).mean(0) for i in probe_idx]
    out["CL13_color_stability"] = float(np.std(cs, 0).mean()) if len(cs) > 1 else np.nan

    # ---- composition from edges and saliency (CO)
    sym, vpc, edge, neg, com, tilt = [], [], [], [], [], []
    for i in probe_idx[:6]:
        g = cv2.cvtColor(fs[i], cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, (320, int(320 * h / w))).astype(np.float32)
        e = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
        sym.append(np.corrcoef(e.ravel(), e[:, ::-1].ravel())[0, 1])
        ed = cv2.Canny(g, 60, 160); edge.append(ed.mean() / 255)
        blk = cv2.resize(ed.astype(np.float32), (16, 9), interpolation=cv2.INTER_AREA); neg.append(np.mean(blk < 255 * .01))
        S = spectral_saliency(g); yy, xx = np.mgrid[0:h, 0:w]
        com.append(np.hypot((S * xx).sum() / S.sum() / w - .5, (S * yy).sum() / S.sum() / h - .5))
        Ls = cv2.HoughLinesP(ed, 1, np.pi / 180, 50, minLineLength=h // 6, maxLineGap=4)
        if Ls is not None:
            ang = np.degrees(np.arctan2(Ls[:, 0, 3] - Ls[:, 0, 1], Ls[:, 0, 2] - Ls[:, 0, 0])) % 180
            v = ang[np.abs(ang - 90) < 15] - 90
            if len(v) >= 3: tilt.append(-np.median(v))
            vp = vanishing(Ls[:, 0].astype(np.float64), w, h)
            if vp is not None: vpc.append(vp)
    out.update(CO2_symmetry=float(np.nanmean(sym)), CO9_edge_density=float(np.mean(edge)), CO7_negative_space=float(np.mean(neg)),
               CO8_saliency_offset=float(np.mean(com)), CA4_roll_deg=float(np.median(tilt)) if tilt else np.nan,
               CO3_vp_converge=float(np.mean([v[0] for v in vpc])) if vpc else np.nan,
               CO3_vp_center=float(np.mean([v[1] for v in vpc])) if vpc else np.nan)
    if h > 0:
        out["CA11_aspect"] = np.nan        # filled at clip level

    # ---- camera and subject motion (CM): similarity transform between consecutive frames on background features
    T, subj, bgm, div = [], [], [], []
    gray = [cv2.cvtColor(cv2.resize(f, (320, int(320 * h / w))), cv2.COLOR_BGR2GRAY) for f in fs]
    sc = 320 / w
    for k in range(len(gray) - 1):
        a, b = gray[k], gray[k + 1]
        fl = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0) * (fps / SFPS if fps else 1)          # displacement per 1/SFPS s, comparable across frame rates
        mag = np.hypot(fl[..., 0], fl[..., 1])
        d = dets[k] if k in dets else faces(fs[k])            # person mask on every frame pair
        pm = body_mask(a.shape, [np.r_[x[:4] * sc, x[4:]] for x in d]) if d else np.zeros(a.shape, bool)
        bgm.append(mag[~pm].mean()); subj.append(mag[pm].mean() if pm.any() else np.nan)
        div.append(float(np.mean(np.gradient(fl[..., 0], axis=1) + np.gradient(fl[..., 1], axis=0))) * 100)
        p0 = cv2.goodFeaturesToTrack(a, 200, .01, 8, mask=(~pm).astype(np.uint8) * 255)
        if p0 is None or len(p0) < 12: T.append([np.nan] * 4); continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
        ok = st.ravel() == 1
        M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], ransacReprojThreshold=2.0) if ok.sum() >= 12 else (None, None)
        if M is None: T.append([np.nan] * 4); continue
        s_ = np.hypot(M[0, 0], M[1, 0]); T.append([M[0, 2] / 320, M[1, 2] / 320, np.log(s_), np.degrees(np.arctan2(M[1, 0], M[0, 0]))])
    T = np.array(T) if T else np.zeros((0, 4))
    if len(T) and np.isfinite(T).any():
        tx, ty, ls, rot = [np.nan_to_num(T[:, j]) for j in range(4)]
        pan, tilt_, zoom = np.abs(tx).mean() * fps, np.abs(ty).mean() * fps, ls.mean() * fps     # per second (fps = analysis rate), frame widths
        if len(tx) >= 5:
            k5 = np.ones(5) / 5; hp = lambda v: v - np.convolve(np.pad(v, 2, mode="edge"), k5, "valid")
            jit = float(np.hypot(hp(tx), hp(ty)).mean() * fps)
        else: jit = np.nan
        speed = np.hypot(tx, ty).mean() * fps
        out.update(CM3_cam_speed=float(speed), CM5_pan_signed=float(tx.mean() * fps), CM5_tilt_signed=float(ty.mean() * fps), CM5_pan=float(pan), CM5_tilt=float(tilt_), CM6_zoom_scale=float(zoom),
                   CM4_shake=float(jit), CM5_roll_rate=float(np.abs(rot).mean() * fps),
                   CM5_horizontal_share=float(pan / (pan + tilt_ + 1e-6)))
        moving = speed > .03 or abs(zoom) > .02
        out["CM2_static"] = bool(not moving)
        kind = "static"
        if moving:
            if jit > .6 * speed and jit > .02: kind = "handheld"
            elif abs(zoom) > max(pan, tilt_) * .8: kind = "push_in" if zoom > 0 else "pull_out"
            elif pan >= tilt_: kind = "pan_or_truck"
            else: kind = "tilt_or_crane"
        out["CM1_move_type"] = kind
    out["CM9_motion"] = float(np.mean(bgm + [x for x in subj if x == x])) if bgm else np.nan
    out["CM8_cam_over_subject"] = float(np.nanmean(bgm) / (np.nanmean(subj) + 1e-3)) if bgm and np.isfinite(np.nanmean(subj)) else np.nan
    out["CM6_divergence"] = float(np.mean(div)) if div else np.nan
    fh = [(i, max(x[3] for x in dets[i]) / h) for i in probe_idx if dets[i]]
    out["CM6_face_growth"] = float((fh[-1][1] - fh[0][1]) / fh[0][1]) if len(fh) >= 2 else np.nan

    # ---- depth of field proxy (CA8): face sharpness over background sharpness
    dof = []
    for i in probe_idx:
        if not dets[i]: continue
        g = cv2.cvtColor(fs[i], cv2.COLOR_BGR2GRAY).astype(np.float32); lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        fm = face_mask(g.shape, dets[i]); bm = ~body_mask(g.shape, dets[i])
        if fm.sum() > 100 and bm.sum() > 1000: dof.append(np.log((lap[fm].mean() + 1e-3) / (lap[bm].mean() + 1e-3)))
    out["CA8_dof_ratio"] = float(np.median(dof)) if dof else np.nan

    # ---- texture on native-resolution frames (TX)
    out.update(texture(natives))
    out["shot_seconds"] = n / fps
    return out, dets


def vanishing(L, w, h):
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
    L = L[(np.abs(ang - 90) > 12) & (np.minimum(ang, 180 - ang) > 8)]
    if len(L) < 4: return None
    pts = []
    for i in range(min(len(L), 40)):
        for j in range(i + 1, min(len(L), 40)):
            x1, y1, x2, y2 = L[i]; x3, y3, x4, y4 = L[j]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-6: continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
            if 0 <= px < w and 0 <= py < h: pts.append((px, py))
    if not pts: return (0.0, np.nan)
    H, xe, ye = np.histogram2d([p[0] for p in pts], [p[1] for p in pts], bins=[12, 8], range=[[0, w], [0, h]])
    i, j = np.unravel_index(H.argmax(), H.shape); vx, vy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    r = .06 * np.hypot(w, h)
    d = np.abs((L[:, 3] - L[:, 1]) * vx - (L[:, 2] - L[:, 0]) * vy + L[:, 2] * L[:, 1] - L[:, 3] * L[:, 0]) / (np.hypot(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0]) + 1e-6)
    return float(np.mean(d < r)), float(1 - np.hypot(vx - w / 2, vy - h / 2) / (np.hypot(w, h) / 2))


TEX_W = 1280         # texture is measured at one common width (720p) so 720p and 1080p sources are comparable (note 25b)
GRAIN_BANDS = np.linspace(12, 245, 9)


def edge_width(Y, max_pts=1500):
    """Marziliano et al. (2002) edge width: at strong near-vertical edges, walk left and right along the row to the
    nearest local extrema of intensity; the distance is the edge width in pixels. Light smoothing keeps grain from
    breaking the walk."""
    Ys = cv2.GaussianBlur(Y, (0, 0), .8); gx = cv2.Sobel(Ys, cv2.CV_32F, 1, 0); gy = cv2.Sobel(Ys, cv2.CV_32F, 0, 1)
    e = cv2.Canny(Ys.astype(np.uint8), 40, 120) > 0
    yy, xx = np.where(e & (np.abs(gx) > 2 * np.abs(gy)) & (np.abs(gx) > 60))
    if len(yy) < 50: return np.nan
    idx = np.random.default_rng(0).choice(len(yy), min(max_pts, len(yy)), replace=False); W_ = Y.shape[1]; ws = []
    for y, x in zip(yy[idx], xx[idx]):
        row = Ys[y]; s = np.sign(gx[y, x]); r = x
        while r + 1 < W_ and r - x < 20 and (row[r + 1] - row[r]) * s > 0: r += 1
        l = x
        while l - 1 >= 0 and x - l < 20 and (row[l] - row[l - 1]) * s > 0: l -= 1
        if abs(row[r] - row[l]) > 20: ws.append(r - l)
    return float(np.median(ws)) if len(ws) >= 30 else np.nan


def texture(natives):
    """Grain, edge width, halation and bloom, measured on frames resized to a common 1280 px width (luma 0..255)."""
    out = {}
    if not natives: return out
    gs, gsz, curve, xc, ew, hal, bloom = [], [], [[] for _ in range(len(GRAIN_BANDS) - 1)], [], [], [], []
    hann = np.outer(np.hanning(64), np.hanning(64)); yy_, xx_ = np.mgrid[-32:32, -32:32]; rr = np.hypot(xx_, yy_); ring = (rr > 1) & (rr < 32)
    for f in natives:
        if f.shape[1] > TEX_W: f = cv2.resize(f, (TEX_W, int(round(TEX_W * f.shape[0] / f.shape[1]))), interpolation=cv2.INTER_AREA)
        Y = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = Y - cv2.GaussianBlur(Y, (0, 0), 1.5)
        loc = cv2.GaussianBlur(Y, (0, 0), 3)
        flat = cv2.blur(np.abs(cv2.Laplacian(loc, cv2.CV_32F)), (15, 15)) < 1.0      # smooth regions only
        flat &= (loc > 12) & (loc < 245)
        ew.append(edge_width(Y))
        if flat.mean() >= .02:
            gs.append(res[flat].std())
            for j in range(len(GRAIN_BANDS) - 1):                       # grain strength against brightness (AV1 style curve)
                m = flat & (loc >= GRAIN_BANDS[j]) & (loc < GRAIN_BANDS[j + 1])
                if m.sum() > 1500: curve[j].append(res[m].std())
            # grain size: Hann-windowed power spectrum averaged over the flattest 64x64 tiles, spectral centroid
            H_, W_ = Y.shape; tiles = []
            for y0 in range(0, H_ - 64, 64):
                for x0 in range(0, W_ - 64, 64):
                    fr = flat[y0:y0 + 64, x0:x0 + 64].mean()
                    if fr > .95: tiles.append((fr, y0, x0))
            if len(tiles) >= 4:
                P = np.zeros((64, 64))
                for _, y0, x0 in sorted(tiles, reverse=True)[:30]:
                    t = Y[y0:y0 + 64, x0:x0 + 64]; t = t - cv2.GaussianBlur(t, (0, 0), 6)        # remove slow shading only
                    P += np.abs(np.fft.fftshift(np.fft.fft2(t * hann))) ** 2
                gsz.append((P[ring] * rr[ring]).sum() / (P[ring].sum() + 1e-9) / 32)            # 1 = finest grain
            if flat.sum() > 2000:
                Bc = cv2.split(f.astype(np.float32)); rs = [c - cv2.GaussianBlur(c, (0, 0), 1.5) for c in Bc]
                xc.append(np.mean([np.corrcoef(rs[a][flat], rs[b][flat])[0, 1] for a, b in [(0, 1), (1, 2), (0, 2)]]))
        hl = (Y > 245).astype(np.uint8)
        if 50 < hl.sum() < .05 * hl.size:
            r1 = cv2.dilate(hl, np.ones((9, 9), np.uint8)) - hl; r3 = cv2.dilate(hl, np.ones((31, 31), np.uint8)) - cv2.dilate(hl, np.ones((21, 21), np.uint8))
            R, G = f[..., 2].astype(np.float32), f[..., 1].astype(np.float32)
            hal.append((R - G)[r1 > 0].mean() - (R - G)[r3 > 0].mean())
            bloom.append(Y[r1 > 0].mean() - Y[r3 > 0].mean())
    med = lambda v: float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else np.nan
    cv_ = [med(c) for c in curve]
    out.update(TX1_grain=med(gs), TX2_grain_size=med(gsz), TX4_grain_chroma_corr=med(xc), TX5_edge_width=med(ew),
               TX6_halation=med(hal), TX7_bloom=med(bloom), TX3_grain_curve=[round(v, 3) if v == v else None for v in cv_])
    sh, mi, hi_ = np.nanmean(cv_[:3]) if np.isfinite(cv_[:3]).any() else np.nan, np.nanmean(cv_[3:6]) if np.isfinite(cv_[3:6]).any() else np.nan, \
        np.nanmean(cv_[6:]) if np.isfinite(cv_[6:]).any() else np.nan
    out.update(TX3_grain_sh=sh, TX3_grain_mid=mi, TX3_grain_hi=hi_, TX3_mid_over_sh=mi / sh if sh == sh and mi == mi and sh > 0 else np.nan)
    return out


# ---------------------------------------------------------------- clip
def encode_info(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 24; n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cc = int(cap.get(cv2.CAP_PROP_FOURCC)); cap.release()
    codec = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
    kbps = os.path.getsize(path) * 8 / (n / fps) / 1000
    return dict(src_height=int(h), src_fps=float(fps), codec=codec, kbps=float(kbps), bits_per_pixel=float(kbps * 1000 / (w * h * fps)))


def process_clip(args):
    path, meta, outdir = args
    vid = meta["videoid"]; dst = os.path.join(outdir, "shots", vid + ".json")
    if os.path.exists(dst): return vid, "skip"
    small, native, fps, thumbs, step = read_clip(path)
    if len(small) < 20: return vid, "too_short"
    s0, s1 = int(2 * fps), len(small) - int(3 * fps)
    ec = [i for i in range(max(s0, s1 - int(60 * fps)), s1) if endcard(small[i])]
    s1 = ec[0] if ec else s1
    fs = small[s0:s1]
    t, b, l, r = crop_box(fs); sc = None
    fs = [f[t:b, l:r] for f in fs]
    aspect = (r - l) / max(1, b - t)
    nat = {}
    for k, f in native.items():
        if s0 <= k < s1:
            sc = f.shape[1] / WORK_W; nat[k - s0] = f[int(t * sc): int(b * sc), int(l * sc): int(r * sc)]
    scenes, gradual = transnet_cuts(thumbs)
    bounds = []
    for st, en in scenes:                                   # native frames -> analysis frames within [s0, s1)
        a, b2 = int(np.ceil(st / step)) - s0, en // step + 1 - s0
        a, b2 = max(a, 0), min(b2, len(fs))
        if b2 - a >= 1: bounds.append((a, b2))
    rows = []
    for a, b2 in bounds:
        if b2 - a < SFPS: continue
        seg = fs[a:b2]; nseg = [nat[k] for k in range(a, b2) if k in nat]
        m, _ = shot_metrics(seg, nseg, fps)
        m["CA11_aspect"] = aspect
        rows.append(dict(videoid=vid, style=meta["style"], title=meta["title"], imdbid=meta["imdbid"], year=meta.get("year"),
                         start=a, end=b2, **m))
    # clip-level editing metrics (ED): from all cuts in the clip, including shots shorter than 1 s
    lens = np.array([(b2 - a) / fps for a, b2 in bounds])
    seq = [(r_["start"], r_["ST2_face_x"], r_["CA1_face_size"], r_["ST1_faces"]) for r_ in rows]
    sr, scale_jump = [], []
    for (s_a, xa, sa, na), (s_b, xb, sb, nb) in zip(seq[:-1], seq[1:]):
        if na and nb and 0.6 < na < 1.5 and 0.6 < nb < 1.5 and xa == xa and xb == xb: sr.append((xa - .5) * (xb - .5) < 0)
        if sa == sa and sb == sb: scale_jump.append(abs(np.log(sb / sa)))
    clip = dict(ED1_asl=float(lens.mean()), ED2_median=float(np.median(lens)), ED2_cv=float(lens.std() / (lens.mean() + 1e-6)),
                ED_n_shots=len(bounds), ED3_gradual_per_min=gradual / (len(fs) / fps / 60), ED4_shot_reverse=float(np.mean(sr)) if sr else np.nan,
                ED6_scale_jump=float(np.mean(scale_jump)) if scale_jump else np.nan, analysed_seconds=len(fs) / fps,
                width=int(small[0].shape[1] * (sc or 1)), aspect=aspect, **encode_info(path))
    json.dump(dict(clip=clip, shots=rows), open(dst, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return vid, f"ok {len(rows)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cliplist"); ap.add_argument("clipdir"); ap.add_argument("outdir")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--ids", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, "shots"), exist_ok=True)
    cl = pd.read_csv(a.cliplist)
    if "year_y" in cl: cl["year"] = cl["year_y"]
    files = {os.path.splitext(x)[0]: os.path.join(a.clipdir, x) for x in os.listdir(a.clipdir) if x.endswith((".mp4", ".webm", ".mkv"))}
    cl = cl[cl.videoid.isin(files)]
    if a.ids: cl = cl[cl.videoid.isin(a.ids.split(","))]
    if a.n: cl = cl.groupby("style").head(max(1, a.n // cl["style"].nunique()))
    jobs = [(files[r.videoid], r._asdict(), a.outdir) for r in cl.itertuples(index=False)]
    print(f"{len(jobs)} clips", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(os.path.join(a.outdir, "clip_errors.txt"), "a") as err:
        futs = {ex.submit(process_clip, j): j[1]["videoid"] for j in jobs}
        for f in as_completed(futs):
            try: print(*f.result(), flush=True)
            except Exception: err.write(futs[f] + "\n" + traceback.format_exc() + "\n"); err.flush(); print(futs[f], "ERROR", flush=True)
