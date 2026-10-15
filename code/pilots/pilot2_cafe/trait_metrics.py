"""Metrics that operationalize each line of traits.json. Works on a list of BGR frames from ONE shot.
Trait -> metric (English names used in all tables):
  cool gray-blue / warm          -> warmth_b      mean CIELAB b* (+ yellow/warm, - blue/cool)
  low saturation / saturated     -> saturation    mean HSV S
  high contrast                  -> contrast      L* p95 - p5
  pale characters, dark bg       -> face_minus_bg mean L* inside faces - median L* outside faces
  warm backlight, light shafts   -> backlight     L* p98 of upper half - mean face L*
  centered symmetry              -> symmetry      correlation of Sobel edge map with its mirror
  one-point perspective          -> vp_converge   share of long lines passing near the dominant vanishing point
                                    vp_center     1 - distance(vanishing point, frame center) / half diagonal
  frontal eye-level view         -> frontal_faces share of detected faces with |yaw proxy| < 0.15
  pastel colors                  -> pastel        share of pixels with L* > 65 and 0.12 < S < 0.45
  large single red/yellow areas  -> red_yellow    share of pixels red or yellow with S > 0.5 and V > 0.35
  slow dolly / push-in           -> push_in       mean optical-flow divergence (> 0 = moving in)
  lateral tracking and pans      -> lateral       |mean horizontal flow| / mean flow magnitude
  push-in to the faces           -> face_growth   relative change of face height across the shot
  long take                      -> shot_seconds  shot length (generated clips are capped at 4-5 s)
Not measurable with these tools (human or VLM rating only): gothic, expressionist design, black-and-white stripes,
real locations, realistic and weighty, IMAX large format (generation is fixed 16:9), wide-angle lens, cold and detached,
meticulous sets, color symbolism, ritualized composition."""
import cv2, numpy as np, os
_YN = None
def faces(f):
    global _YN
    if _YN is None:
        _YN = cv2.FaceDetectorYN.create(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yunet.onnx"), "", (320, 320), 0.7)
    _YN.setInputSize((f.shape[1], f.shape[0])); _, d = _YN.detect(f)
    return [] if d is None else [x for x in d if x[3] >= 0.08 * f.shape[0]]

def yaw_proxy(d):
    re_, le, nose = d[4:6], d[6:8], d[8:10]
    eye_mid = (re_ + le) / 2; ed = np.linalg.norm(le - re_) + 1e-6
    return (nose[0] - eye_mid[0]) / ed

def vanishing(gray):
    e = cv2.Canny(gray, 60, 160)
    L = cv2.HoughLinesP(e, 1, np.pi / 180, 60, minLineLength=gray.shape[1] // 10, maxLineGap=6)
    if L is None or len(L) < 6: return np.nan, np.nan
    L = L[:, 0].astype(np.float64)
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
    L = L[(np.abs(ang - 90) > 12) & (np.minimum(ang, 180 - ang) > 8)]    # drop near-vertical and near-horizontal
    if len(L) < 4: return np.nan, np.nan
    h, w = gray.shape; pts = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            x1, y1, x2, y2 = L[i]; x3, y3, x4, y4 = L[j]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-6: continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
            if 0 <= px < w and 0 <= py < h: pts.append((px, py))
    if not pts: return 0.0, np.nan
    H, xe, ye = np.histogram2d([p[0] for p in pts], [p[1] for p in pts], bins=[12, 8], range=[[0, w], [0, h]])
    i, j = np.unravel_index(H.argmax(), H.shape); vx, vy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    r = 0.06 * np.hypot(w, h)
    d = np.abs((L[:, 3] - L[:, 1]) * vx - (L[:, 2] - L[:, 0]) * vy + L[:, 2] * L[:, 1] - L[:, 3] * L[:, 0]) / (np.hypot(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0]) + 1e-6)
    return float(np.mean(d < r)), float(1 - np.hypot(vx - w / 2, vy - h / 2) / (np.hypot(w, h) / 2))

def shot_metrics(fs, fps):
    """fs: frames of one shot at ~fps frames per second (BGR, letterbox already removed)."""
    out = {}
    lab = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in fs]
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32) for f in fs]
    Ls = [x[..., 0] * 100 / 255 for x in lab]
    out["warmth_b"] = float(np.mean([x[..., 2].mean() - 128 for x in lab]))
    out["saturation"] = float(np.mean([x[..., 1].mean() / 255 for x in hsv]))
    allL = np.concatenate([l.ravel()[::7] for l in Ls]); out["contrast"] = float(np.percentile(allL, 95) - np.percentile(allL, 5))
    out["pastel"] = float(np.mean([np.mean((l > 65) & (h[..., 1] / 255 > .12) & (h[..., 1] / 255 < .45)) for l, h in zip(Ls, hsv)]))
    out["red_yellow"] = float(np.mean([np.mean(((h[..., 0] < 10) | (h[..., 0] > 170) | ((h[..., 0] > 18) & (h[..., 0] < 35)))
                                               & (h[..., 1] > 127) & (h[..., 2] > 89)) for h in hsv]))
    sym = []
    for f in fs[:: max(1, len(fs) // 6)]:
        g = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        sym.append(np.corrcoef(e.ravel(), e[:, ::-1].ravel())[0, 1])
    out["symmetry"] = float(np.mean(sym))
    vp = [vanishing(cv2.cvtColor(cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))), cv2.COLOR_BGR2GRAY)) for f in fs[:: max(1, len(fs) // 4)]]
    out["vp_converge"] = float(np.nanmean([v[0] for v in vp])) if any(np.isfinite(v[0]) for v in vp) else np.nan
    out["vp_center"] = float(np.nanmean([v[1] for v in vp])) if any(np.isfinite(v[1]) for v in vp) else np.nan
    fr, fh, fb, bl, yaws = [], [], [], [], []
    for k, (f, l) in enumerate(zip(fs, Ls)):
        if k % max(1, len(fs) // 8): continue
        dd = faces(f); fr.append(len(dd))
        if not dd: continue
        mask = np.zeros(l.shape, bool)
        for x in dd:
            x0, y0, w0, h0 = [int(v) for v in x[:4]]; mask[max(0, y0):y0 + h0, max(0, x0):x0 + w0] = True
            yaws.append(abs(yaw_proxy(x)))
        fmean = l[mask].mean(); fb.append(fmean - np.median(l[~mask]))
        bl.append(np.percentile(l[: l.shape[0] // 2], 98) - fmean)
        fh.append((k, max(x[3] for x in dd) / f.shape[0]))
    out["faces_per_frame"] = float(np.mean(fr)) if fr else 0.0
    out["face_minus_bg"] = float(np.mean(fb)) if fb else np.nan
    out["backlight"] = float(np.mean(bl)) if bl else np.nan
    out["frontal_faces"] = float(np.mean(np.array(yaws) < .15)) if yaws else np.nan
    out["face_growth"] = float((fh[-1][1] - fh[0][1]) / fh[0][1]) if len(fh) >= 2 else np.nan
    g = [cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY) for f in fs]
    div, lat = [], []
    for a, b in zip(g[:-1], g[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0) * (5.0 / fps)   # per 1/5 s
        u, v = fl[..., 0], fl[..., 1]
        div.append(float(np.mean(np.gradient(u, axis=1) + np.gradient(v, axis=0))) * 100)
        mag = np.hypot(u, v).mean(); lat.append(abs(u.mean()) / (mag + 1e-6) if mag > .05 else 0.0)
    out["push_in"] = float(np.mean(div)) if div else np.nan
    out["lateral"] = float(np.mean(lat)) if lat else np.nan
    out["shot_seconds"] = len(fs) / fps
    return out
