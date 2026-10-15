"""Which metrics carry style, and at which level? Reads extract.py output (OUTDIR/shots/*.json).
For every numeric shot metric:
  1. variance split at shot level: between director, between film within director, within film;
  2. director effect: Kruskal-Wallis on film means across directors (film = unit), epsilon squared, BH q;
  3. film effect: Kruskal-Wallis on shot values across films, within each director, combined by median epsilon squared;
  4. bitrate confound: Spearman correlation of film means with log bits per pixel;
  5. redundancy: clusters of metrics with |Spearman| > 0.9 on film means.
Multivariate: leave-one-film-out director classification (chance 1/k), with all metrics, per category, and with
encoding covariates only (bitrate, codec, resolution) as a confound baseline.
Optionally merges direction-test results (perturb sensitivity.csv) into the final per-metric verdict.

python discriminate.py EXTRACT_OUTDIR OUTDIR [--perturb sensitivity.csv] [--dialog-only]"""
import os, sys, json, glob, argparse, numpy as np, pandas as pd
from scipy.stats import kruskal, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ap = argparse.ArgumentParser(); ap.add_argument("indir"); ap.add_argument("outdir")
ap.add_argument("--perturb", default=""); ap.add_argument("--control-encoding", action="store_true"); ap.add_argument("--dialog-only", action="store_true")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

rows = []
for f in glob.glob(os.path.join(a.indir, "shots", "*.json")):
    d = json.load(open(f)); c = d["clip"]
    for s in d["shots"]:
        s.update({f"clip_{k}": v for k, v in c.items()}); rows.append(s)
df = pd.DataFrame(rows)
df = df[~df["black"].astype(bool)]
if a.dialog_only: df = df[df["dialog"].astype(bool)]
df["film"] = df["title"].astype(str) + " (" + df["year"].astype(str) + ")"
df["log_bpp"] = np.log(df["clip_bits_per_pixel"].astype(float))

df["codec_vp9"] = df["clip_codec"].astype(str).str.startswith("VP").astype(float); df["src_h"] = df["clip_src_height"].astype(float)
ENC = ["log_bpp", "codec_vp9", "src_h"]
SKIP = {"start", "end", "year", "shot_seconds", "dialog", "black", "CA11_aspect"}
metrics = [c for c in df.columns if c[:2] in ("ED", "CM", "CA", "CO", "ST", "LI", "CL", "GR", "TX") and c not in SKIP
           and pd.api.types.is_numeric_dtype(df[c])]
clip_metrics = [c for c in df.columns if c.startswith("clip_ED") and pd.api.types.is_numeric_dtype(df[c])]
if a.control_encoding:
    Xe = np.column_stack([np.ones(len(df))] + [df[c].values for c in ENC])
    for m in metrics:
        y = df[m].replace([np.inf, -np.inf], np.nan).values.astype(float); ok = np.isfinite(y) & np.isfinite(Xe).all(1)
        if ok.sum() > 50:
            beta, *_ = np.linalg.lstsq(Xe[ok], y[ok], rcond=None); r = np.full(len(y), np.nan); r[ok] = y[ok] - Xe[ok] @ beta + y[ok].mean()
            df[m] = r
CAT = {"ED": "editing", "CM": "camera_motion", "CA": "camera_optics", "CO": "composition", "ST": "staging",
       "LI": "lighting", "CL": "color", "GR": "tone_grading", "TX": "texture"}
k_dir = df["style"].nunique()

res = []
for m in metrics:
    d = df[["style", "film", "log_bpp", m]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 50 or d[m].std() == 0: continue
    gm = d[m].mean(); tot = ((d[m] - gm) ** 2).sum()
    dm = d.groupby("style")[m].transform("mean"); fm_ = d.groupby("film")[m].transform("mean")
    v_dir = ((dm - gm) ** 2).sum() / tot; v_film = ((fm_ - dm) ** 2).sum() / tot
    fm = d.groupby(["style", "film"]).agg(v=(m, "mean"), bpp=("log_bpp", "mean")).reset_index()
    groups = [g["v"].values for _, g in fm.groupby("style") if len(g)]
    H, p = kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan); n = len(fm)
    eps_dir = max(0.0, (H - len(groups) + 1) / (n - len(groups))) if n > len(groups) else np.nan
    eps_film = []
    for _, g in d.groupby("style"):
        fg = [x[m].values for _, x in g.groupby("film") if len(x) >= 5]
        if len(fg) >= 2:
            Hf, _ = kruskal(*fg); nf = sum(len(x) for x in fg); eps_film.append(max(0.0, (Hf - len(fg) + 1) / (nf - len(fg))))
    rho_bpp = spearmanr(fm["v"], fm["bpp"])[0] if fm["bpp"].nunique() > 2 else np.nan
    res.append(dict(category=CAT[m[:2]], metric=m, shots=len(d), films=n, nan_share=float(df[m].isna().mean()),
                    var_director=v_dir, var_film=v_film, var_shot=1 - v_dir - v_film,
                    eps2_director=eps_dir, p_director=p, eps2_film_within_dir=float(np.median(eps_film)) if eps_film else np.nan,
                    rho_bitrate=rho_bpp))
res = pd.DataFrame(res)
res["q_director"] = multipletests(res["p_director"].fillna(1), method="fdr_bh")[1]

# redundancy on film means
FM = df.groupby("film")[list(res.metric)].mean()
R = FM.corr(method="spearman").abs()
cluster, seen = {}, set()
order = res.sort_values("eps2_director", ascending=False).metric.tolist()
for m in order:
    if m in seen: continue
    grp = [x for x in order if x not in seen and R.loc[m, x] > 0.9]
    for x in grp: seen.add(x); cluster[x] = m
res["redundant_with"] = res.metric.map(lambda x: "" if cluster.get(x) == x else cluster.get(x, ""))

# direction test
if a.perturb and os.path.exists(a.perturb):
    pt = pd.read_csv(a.perturb).groupby("target").agg(direction_monotonic=("monotonic", "max"), direction_right=("right_direction_l3", "max"))
    res = res.merge(pt, left_on="metric", right_index=True, how="left")
else:
    res["direction_monotonic"] = np.nan

def verdict(r):
    disc = r.q_director < .05 and r.eps2_director >= .06 or (r.eps2_film_within_dir == r.eps2_film_within_dir and r.eps2_film_within_dir >= .06)
    conf = abs(r.rho_bitrate) >= .5 if r.rho_bitrate == r.rho_bitrate else False
    tested = r.direction_monotonic == r.direction_monotonic
    if r.redundant_with: return "redundant"
    if r.nan_share > .8: return "too_sparse"
    if disc and not conf and (not tested or r.direction_monotonic >= .8): return "score" if tested else "score_pending_direction_test"
    if disc and conf: return "bitrate_confounded"
    if disc: return "reference"
    return "not_discriminative"
res["verdict"] = res.apply(verdict, axis=1)
res = res.sort_values(["category", "eps2_director"], ascending=[True, False])

# leave-one-film-out director classification
def lofo(cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float); y = df["style"].values; films = df["film"].values
    pred = np.empty(len(df), dtype=object)
    for f in np.unique(films):
        te = films == f
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=.3, class_weight="balanced"))
        clf.fit(X[~te], y[~te]); pred[te] = clf.predict(X[te])
    bal = np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])
    fv = pd.DataFrame(dict(film=films, y=y, p=pred)).groupby("film").agg(y=("y", "first"), p=("p", lambda s: s.mode()[0]))
    return dict(shot_balanced_acc=bal, film_acc=float(np.mean(fv.y == fv.p)), films=len(fv))
clf = [dict(features="encoding_only (confound baseline)", n=3, **lofo(["log_bpp", "codec_vp9", "src_h"]))]
for cat in sorted(set(res.category)):
    cols = res[(res.category == cat) & (res.redundant_with == "")].metric.tolist()
    if cols: clf.append(dict(features=cat, n=len(cols), **lofo(cols)))
allc = res[res.redundant_with == ""].metric.tolist()
clf.append(dict(features="all", n=len(allc), **lofo(allc)))
clf.append(dict(features="all + encoding", n=len(allc) + 3, **lofo(allc + ENC)))
clf = pd.DataFrame(clf); clf["chance"] = 1 / k_dir

res.to_csv(os.path.join(a.outdir, "metric_verdicts.csv"), index=False); clf.to_csv(os.path.join(a.outdir, "lofo_director.csv"), index=False)
df.groupby(["style", "film"])[list(res.metric)].mean().round(4).to_csv(os.path.join(a.outdir, "film_means.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df.groupby("style").agg(shots=("film", "size"), films=("film", "nunique"), clips=("videoid", "nunique")).to_string()); print()
print(res.round(3).to_string(index=False)); print(); print(clf.round(3).to_string(index=False)); print()
print(res.verdict.value_counts().to_string())
