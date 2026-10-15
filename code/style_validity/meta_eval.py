"""Meta-evaluation of style-distance scoring methods, with real films as known answers, then scoring of generated clips.

Scoring methods d(S, R) between a query shot set S and a reference shot set R (lower = closer), on standardized features:
  mean_z    mean over features of |mean_S - mean_R| / pooled sd
  w1        mean over features of the 1D Wasserstein distance
  ks        mean over features of the Kolmogorov-Smirnov statistic
  energy    multivariate energy distance (Szekely and Rizzo)
  mmd       RBF maximum mean discrepancy, median-heuristic bandwidth
  frechet   Frechet distance between Gaussians fitted on PCA(10) features, Ledoit-Wolf covariance
  c2st      cross-validated logistic-regression accuracy separating S from R (0.5 = indistinguishable)
  knn_prec  1 - k-NN precision (share of S outside R's k-NN manifold; Kynkaanniemi et al. 2019)
Known-answer tests (real films only):
  T1 identification: own film (or own director, leave-one-film-out) should be the closest reference; n = 8, 16, 32
  T2 mixture: replacing a growing share of a film's shots with another film's shots should increase the distance
  T3 diversity collapse: a near-duplicate query set (one exemplar and its nearest neighbours) should look farther
  T4 encoding: reference from low-bitrate clips, query from high-bitrate clips of the same film
Generated clips: distance to each director, gain over the no-style control, and percentile against real-vs-real distances.

python meta_eval.py REAL_EXTRACT GEN_EXTRACT GEN_LIST OUTDIR [--verdicts metric_verdicts.csv] [--perturb sensitivity.csv]"""
import os, sys, json, glob, argparse, warnings, numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, spearmanr
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.linalg import sqrtm
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("real"); ap.add_argument("gen"); ap.add_argument("genlist"); ap.add_argument("outdir")
ap.add_argument("--verdicts", default=""); ap.add_argument("--perturb", default=""); ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--all-shots", action="store_true"); ap.add_argument("--quick", action="store_true")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
RNG = np.random.default_rng(0)


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "shots", "*.json")):
        j = json.load(open(f))
        for s in j["shots"]:
            s.update({f"clip_{k}": v for k, v in j["clip"].items()}); rows.append(s)
    return pd.DataFrame(rows)


R = load(a.real); G = load(a.gen)
R = R[~R["black"].astype(bool)].copy(); G = G[~G["black"].astype(bool)].copy()
if not a.all_shots: R = R[R["dialog"].astype(bool)].copy()
R["film"] = R["title"].astype(str) + " (" + R["year"].astype(str) + ")"
gl = pd.read_csv(a.genlist)[["videoid", "condition", "seed"]]
G = G.merge(gl, on="videoid", how="left"); G["model"] = G["title"]; G["content"] = G["imdbid"]
for D in (R, G):
    D["log_bpp"] = np.log(D["clip_bits_per_pixel"].astype(float))
    D["codec_vp9"] = D["clip_codec"].astype(str).str.upper().str.startswith("VP").astype(float)
    D["src_h"] = D["clip_src_height"].astype(float)
ENC = ["log_bpp", "codec_vp9", "src_h"]

# ---------------------------------------------------------------- feature sets
cand = [c for c in R.columns if c[:2] in ("CM", "CA", "CO", "ST", "LI", "CL", "GR", "TX") and pd.api.types.is_numeric_dtype(R[c])
        and c in G.columns and not c.startswith("CA11") and R[c].notna().mean() > .5 and G[c].notna().mean() > .3 and R[c].std() > 0]
sets = {"all": cand}
if a.verdicts and os.path.exists(a.verdicts):
    v = pd.read_csv(a.verdicts)
    sets["screened"] = [m for m in cand if m in set(v[v.verdict.isin(["score", "score_pending_direction_test"])].metric)]
if a.perturb and os.path.exists(a.perturb):
    p = pd.read_csv(a.perturb); ok = set(p[(p.monotonic >= .8) & (p.right_direction_l3 >= .9)].target)
    sets["direction_tested"] = [m for m in cand if m in ok]
CATS = {"color_tone": ("CL", "GR"), "camera_composition_staging": ("CM", "CA", "CO", "ST"), "lighting_texture": ("LI", "TX")}
for k_, pre in CATS.items(): sets[f"only_{k_}"] = [m for m in cand if m.startswith(pre)]
sets = {k: v for k, v in sets.items() if len(v) >= 3}


def prepare(cols, residualize):
    Xr = R[cols].replace([np.inf, -np.inf], np.nan).astype(float).copy(); Xg = G[cols].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    if residualize:
        E = np.column_stack([np.ones(len(R))] + [R[c].values for c in ENC]); Eg = np.column_stack([np.ones(len(G))] + [G[c].values for c in ENC])
        for c in cols:
            y = Xr[c].values; ok = np.isfinite(y)
            b, *_ = np.linalg.lstsq(E[ok], y[ok], rcond=None)
            Xr[c] = y - E @ b + np.nanmean(y); Xg[c] = Xg[c].values - Eg @ b + np.nanmean(y)
    med, mu, sd = Xr.median(), Xr.mean(), Xr.std().replace(0, 1)
    Xr, Xg = Xr.fillna(med), Xg.fillna(med)
    return ((Xr - mu) / sd).values, ((Xg - mu) / sd).values


# ---------------------------------------------------------------- distances
def d_mean_z(S, Rr): return float(np.mean(np.abs(S.mean(0) - Rr.mean(0))))
def d_w1(S, Rr): return float(np.mean([wasserstein_distance(S[:, j], Rr[:, j]) for j in range(S.shape[1])]))
def d_ks(S, Rr): return float(np.mean([ks_2samp(S[:, j], Rr[:, j]).statistic for j in range(S.shape[1])]))
def _pd(A, B): return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0))
def d_energy(S, Rr):
    Rr = Rr if len(Rr) <= 400 else Rr[RNG.choice(len(Rr), 400, replace=False)]
    return float(2 * _pd(S, Rr).mean() - _pd(S, S).mean() - _pd(Rr, Rr).mean())
def d_mmd(S, Rr):
    Rr = Rr if len(Rr) <= 400 else Rr[RNG.choice(len(Rr), 400, replace=False)]
    Z = np.vstack([S, Rr]); D2 = _pd(Z, Z) ** 2; h = np.median(D2[D2 > 0]); K = np.exp(-D2 / h); n = len(S)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())
PCA_FIT = {}
def d_frechet(S, Rr, key):
    pca = PCA_FIT[key]; s, r = pca.transform(S), pca.transform(Rr)
    cs, cr = LedoitWolf().fit(s).covariance_, LedoitWolf().fit(r).covariance_
    cm = sqrtm(cs @ cr).real
    return float(((s.mean(0) - r.mean(0)) ** 2).sum() + np.trace(cs + cr - 2 * cm))
def d_c2st(S, Rr):
    Rr = Rr if len(Rr) <= 4 * len(S) else Rr[RNG.choice(len(Rr), 4 * len(S), replace=False)]
    X = np.vstack([S, Rr]); y = np.r_[np.ones(len(S)), np.zeros(len(Rr))]
    cv = StratifiedKFold(min(4, len(S)), shuffle=True, random_state=0)
    return float(cross_val_score(LogisticRegression(C=.3, max_iter=500, class_weight="balanced"), X, y, cv=cv, scoring="balanced_accuracy").mean())
def d_knn_prec(S, Rr, k=3):
    Rr = Rr if len(Rr) <= 400 else Rr[RNG.choice(len(Rr), 400, replace=False)]
    DR = _pd(Rr, Rr); rad = np.sort(DR, 1)[:, k]; inside = (_pd(S, Rr) <= rad[None, :]).any(1)
    return float(1 - inside.mean())
METHODS = {"mean_z": d_mean_z, "w1": d_w1, "ks": d_ks, "energy": d_energy, "mmd": d_mmd, "frechet": None, "c2st": d_c2st, "knn_prec": d_knn_prec}
if a.quick: METHODS.pop("c2st")


def dist(m, S, Rr, key):
    return d_frechet(S, Rr, key) if m == "frechet" else METHODS[m](S, Rr)


# ---------------------------------------------------------------- known-answer tests
def split_clips(df_f, rng):
    clips = df_f["videoid"].unique(); rng.shuffle(clips); h = len(clips) // 2
    return set(clips[:h]), set(clips[h:])


def run_tests(Xr, key):
    rows = []
    films = R.groupby("film").agg(n=("videoid", "nunique"), shots=("videoid", "size"), d=("style", "first"))
    films = films[(films.n >= 4) & (films.shots >= 40)]
    fidx = {f: np.where(R["film"].values == f)[0] for f in films.index}
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        ref, qry = {}, {}
        for f, ix in fidx.items():
            A, B = split_clips(R.iloc[ix], rng)
            ref[f] = ix[np.isin(R["videoid"].values[ix], list(A))]
            qry[f] = ix[np.isin(R["videoid"].values[ix], list(B))]
        for n in (8, 16, 32):
            for f in films.index:
                if len(qry[f]) < n or len(ref[f]) < 10: continue
                S = Xr[rng.choice(qry[f], n, replace=False)]
                # T1 film identification: own reference must be closest among all film references
                others = [g for g in films.index if g != f and len(ref[g]) >= 10]
                cmp_ = others if len(others) <= 12 else list(rng.choice(others, 12, replace=False))
                for m in METHODS:
                    if m == "c2st" and n != 16: continue
                    own = dist(m, S, Xr[ref[f]], key); oth = [dist(m, S, Xr[ref[g]], key) for g in cmp_]
                    rows.append(dict(test="T1_film", method=m, n=n, rep=rep, film=f, correct=float(own < min(oth)),
                                     auc=float(np.mean([own < o for o in oth])), k=len(oth) + 1))
                # T1 director identification, leave this film out of its director's reference
                d0 = films.loc[f, "d"]; drefs = {}
                for d in R["style"].unique():
                    pool = np.where((R["style"].values == d) & (R["film"].values != f))[0]
                    if len(pool) >= 20: drefs[d] = pool if len(pool) <= 600 else rng.choice(pool, 600, replace=False)
                if d0 in drefs:
                    for m in METHODS:
                        if m == "c2st" and n != 16: continue
                        ds = {d: dist(m, S, Xr[ix], key) for d, ix in drefs.items()}
                        own = ds[d0]; oth = [v for d, v in ds.items() if d != d0]
                        rows.append(dict(test="T1_director", method=m, n=n, rep=rep, film=f, correct=float(own < min(oth)),
                                         auc=float(np.mean([own < o for o in oth])), k=len(ds)))
            if n != 16: continue
            for f in films.index:
                if len(qry[f]) < 16 or len(ref[f]) < 10: continue
                g = rng.choice([x for x in films.index if x != f and films.loc[x, "d"] != films.loc[f, "d"]])
                # T2 mixture: share p of own shots, rest from another director's film
                for m in METHODS:
                    vals = []
                    for p in (1, .75, .5, .25, 0):
                        k_own = int(round(16 * p))
                        S = Xr[np.r_[rng.choice(qry[f], k_own, replace=False), rng.choice(fidx[g], 16 - k_own, replace=False)].astype(int)]
                        vals.append(dist(m, S, Xr[ref[f]], key))
                    rho = spearmanr([1, .75, .5, .25, 0], vals)[0]
                    rows.append(dict(test="T2_mixture", method=m, n=16, rep=rep, film=f, rho=float(-rho), monotone=float(np.all(np.diff(vals) >= 0))))
                # T3 diversity collapse: one exemplar and its 15 nearest neighbours vs a random draw of 16
                pool = Xr[qry[f]]; e = rng.integers(len(pool)); nn = np.argsort(((pool - pool[e]) ** 2).sum(1))[:16]
                rnd = rng.choice(len(pool), 16, replace=False)
                for m in METHODS:
                    rows.append(dict(test="T3_collapse", method=m, n=16, rep=rep, film=f,
                                     correct=float(dist(m, pool[nn], Xr[ref[f]], key) > dist(m, pool[rnd], Xr[ref[f]], key))))
        # T4 encoding: low-bitrate clips as reference, high-bitrate clips as query, film identification at n = 16
        bpp = R.groupby("videoid")["clip_bits_per_pixel"].first()
        lo, hi = {}, {}
        for f, ix in fidx.items():
            v = R["videoid"].values[ix]; cl = pd.Series(bpp[np.unique(v)]).sort_values()
            if len(cl) < 4 or cl.iloc[-1] / cl.iloc[0] < 1.5: continue
            h = len(cl) // 2; lo[f] = ix[np.isin(v, cl.index[:h])]; hi[f] = ix[np.isin(v, cl.index[h:])]
        for f in lo:
            if len(hi[f]) < 16 or len(lo[f]) < 10: continue
            S = Xr[rng.choice(hi[f], 16, replace=False)]
            others = [g for g in films.index if g != f and len(ref[g]) >= 10]
            cmp_ = others if len(others) <= 12 else list(rng.choice(others, 12, replace=False))
            for m in METHODS:
                own = dist(m, S, Xr[lo[f]], key); oth = [dist(m, S, Xr[ref[g]], key) for g in cmp_]
                rows.append(dict(test="T4_encoding", method=m, n=16, rep=rep, film=f, correct=float(own < min(oth)), auc=float(np.mean([own < o for o in oth]))))
    return pd.DataFrame(rows), films


def eval_config(arg):
    sname, resid = arg; cols = sets[sname]; key = f"{sname}|{'resid' if resid else 'raw'}"
    Xr, _ = prepare(cols, resid); PCA_FIT[key] = PCA(min(10, len(cols)), random_state=0).fit(Xr)
    df_, fi = run_tests(Xr, key)
    df_["features"] = sname; df_["encoding"] = "residualized" if resid else "raw"; df_["n_features"] = len(cols)
    print("done", key, len(df_), flush=True); return df_, fi


from concurrent.futures import ProcessPoolExecutor
configs = [(s_, r_) for s_ in sets for r_ in (False, True)]
with ProcessPoolExecutor(min(len(configs), int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))) as ex:
    res_ = list(ex.map(eval_config, configs))
T = pd.concat([r_[0] for r_ in res_]); films_info = res_[0][1]
T.to_csv(os.path.join(a.outdir, "meta_tests_raw.csv"), index=False)

# ---------------------------------------------------------------- summary of known-answer tests
def summ(t):
    g = t.groupby(["features", "encoding", "method"])
    out = pd.DataFrame({
        "film_id_n8": t[(t.test == "T1_film") & (t.n == 8)].groupby(["features", "encoding", "method"]).correct.mean(),
        "film_id_n16": t[(t.test == "T1_film") & (t.n == 16)].groupby(["features", "encoding", "method"]).correct.mean(),
        "film_id_n32": t[(t.test == "T1_film") & (t.n == 32)].groupby(["features", "encoding", "method"]).correct.mean(),
        "film_auc_n16": t[(t.test == "T1_film") & (t.n == 16)].groupby(["features", "encoding", "method"]).auc.mean(),
        "director_id_n16": t[(t.test == "T1_director") & (t.n == 16)].groupby(["features", "encoding", "method"]).correct.mean(),
        "director_auc_n16": t[(t.test == "T1_director") & (t.n == 16)].groupby(["features", "encoding", "method"]).auc.mean(),
        "mixture_rho": t[t.test == "T2_mixture"].groupby(["features", "encoding", "method"]).rho.mean(),
        "mixture_monotone": t[t.test == "T2_mixture"].groupby(["features", "encoding", "method"]).monotone.mean(),
        "collapse_detect": t[t.test == "T3_collapse"].groupby(["features", "encoding", "method"]).correct.mean(),
        "encoding_id_n16": t[t.test == "T4_encoding"].groupby(["features", "encoding", "method"]).correct.mean(),
    })
    k_f = t[t.test == "T1_film"].k.median(); k_d = t[t.test == "T1_director"].k.median()
    out["film_chance"] = 1 / k_f; out["director_chance"] = 1 / k_d
    out["overall"] = out[["film_auc_n16", "director_auc_n16", "mixture_rho", "collapse_detect"]].mean(1)
    return out.reset_index().sort_values("overall", ascending=False)
S_ = summ(T); S_.to_csv(os.path.join(a.outdir, "meta_summary.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
print(S_.round(3).to_string(index=False), flush=True)

# ---------------------------------------------------------------- generated clips
best = S_.iloc[0]; key = f"{best.features}|{'resid' if best.encoding == 'residualized' else 'raw'}"
methods_g = list(dict.fromkeys([best.method, "energy", "w1", "mean_z", "knn_prec"]))
cols = sets[best.features]; Xr, Xg = prepare(cols, best.encoding == "residualized"); PCA_FIT[key] = PCA(min(10, len(cols)), random_state=0).fit(Xr)
dirs = sorted(R["style"].unique()); dref = {d: np.where(R["style"].values == d)[0] for d in dirs}
rows = []
for (model, cond, sty), g in G.groupby(["model", "condition", "style"]):
    S = Xg[np.where((G["model"].values == model) & (G["condition"].values == cond) & (G["style"].values == sty))[0]]
    if len(S) < 2: continue
    for m in methods_g:
        if m == "c2st" and len(S) < 4: continue
        ds = {d: dist(m, S, Xr[dref[d]], key) for d in dirs}
        rows.append(dict(model=model, condition=cond, style=sty, n_shots=len(S), method=m, **{f"d_{d}": v for d, v in ds.items()},
                         target_rank=(sorted(ds.values()).index(ds[sty]) + 1) if sty in ds else np.nan))
GD = pd.DataFrame(rows)
# style gain: distance to target director for the styled set minus for the no-style control of the same model
gain = []
for (model, m), g in GD.groupby(["model", "method"]):
    ctrl = g[g.condition == "none"]
    if ctrl.empty: continue
    for _, r in g[g.condition != "none"].iterrows():
        if f"d_{r['style']}" in ctrl: gain.append(dict(model=model, method=m, condition=r.condition, style=r["style"], n_shots=r.n_shots,
                                                       gain=float(ctrl[f"d_{r['style']}"].iloc[0] - r[f"d_{r['style']}"]), target_rank=r.target_rank))
GG = pd.DataFrame(gain)
# realism gap: distance of each model's full generated set to the pooled real set, against real held-out sets of the same size
null = []
for rep in range(40):
    rng = np.random.default_rng(100 + rep); f = rng.choice(R["film"].unique()); ix = np.where(R["film"].values == f)[0]
    rest = np.where(R["film"].values != f)[0]
    for n in (16, 32):
        if len(ix) >= n: null.append(dict(n=n, **{m: dist(m, Xr[rng.choice(ix, n, replace=False)], Xr[rest], key) for m in methods_g if m != "c2st" or n >= 8}))
NULL = pd.DataFrame(null); gap = []
for model, g in G.groupby("model"):
    ix = np.where(G["model"] == model)[0]
    for n in (16, 32):
        if len(ix) < n: continue
        S = Xg[np.random.default_rng(7).choice(ix, n, replace=False)]
        for m in methods_g:
            dv = dist(m, S, Xr, key); nv = NULL[NULL.n == n][m].dropna()
            gap.append(dict(model=model, n=n, method=m, distance=dv, real_median=nv.median(), real_p95=nv.quantile(.95), pct_of_real=float((nv < dv).mean() * 100)))
GP = pd.DataFrame(gap)
GD.to_csv(os.path.join(a.outdir, "gen_distances.csv"), index=False); GG.to_csv(os.path.join(a.outdir, "gen_style_gain.csv"), index=False)
GP.to_csv(os.path.join(a.outdir, "gen_realism_gap.csv"), index=False)
json.dump(dict(best=dict(best), feature_sets={k: v for k, v in sets.items()}, films=films_info.reset_index().to_dict("records")),
          open(os.path.join(a.outdir, "meta_info.json"), "w"), indent=1, default=str)
print(); print("BEST", dict(best)); print(GP.round(3).to_string(index=False)); print()
if not GG.empty:
    print(GG.groupby(["method", "condition"]).agg(mean_gain=("gain", "mean"), share_positive=("gain", lambda x: (x > 0).mean()),
                                                  target_top1=("target_rank", lambda x: (x == 1).mean()), sets=("gain", "size")).round(3).to_string())
