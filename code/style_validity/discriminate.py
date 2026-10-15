"""Can each metric tell directors apart? Film is the unit of analysis.
1. Per metric: Kruskal-Wallis across directors on film means, epsilon squared, Benjamini-Hochberg q.
2. Per metric: variance split into between director, between film within director, within film (shot level).
3. Multivariate: leave-one-film-out classification of director from shot metrics (chance = 1/6).
python discriminate.py real_shots_ext.json OUTDIR"""
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

IN, OUT = sys.argv[1], sys.argv[2]
df = pd.DataFrame(json.load(open(IN)))
GROUPS = {
    "tone": ["luma", "contrast", "black_p2", "white_p98", "clip_hi"],
    "color": ["warmth_b", "tint_a", "chroma", "saturation", "pastel", "red_yellow"],
    "grading": ["neutral_b_sh", "neutral_b_mid", "neutral_b_hi", "neutral_a_sh", "neutral_a_hi", "split_b", "split_a",
                "lum_explained_b", "lum_explained_a", "hi_desat", "skin_hue", "skin_chroma", "vignette"],
    "staging": ["faces_per_frame", "face_size", "face_offcenter", "frontal_faces", "face_minus_bg", "backlight"],
    "composition": ["symmetry", "vp_converge", "vp_center"],
    "camera_editing": ["motion", "push_in", "lateral", "face_growth", "shot_seconds"],
}
METRICS = [m for g in GROUPS.values() for m in g]
GROUP_OF = {m: g for g, ms in GROUPS.items() for m in ms}
df["film"] = df["title"]
k = df["style"].nunique()

rows = []
for m in METRICS:
    d = df[["style", "film", m]].dropna()
    fm = d.groupby(["style", "film"])[m].mean().reset_index()
    samples = [g[m].values for _, g in fm.groupby("style")]
    H, p = kruskal(*samples)
    n = len(fm); eps2 = max(0.0, (H - k + 1) / (n - k))
    # variance components at shot level (sums of squares, nested: director > film > shot)
    gm = d[m].mean(); tot = ((d[m] - gm) ** 2).sum()
    dmean = d.groupby("style")[m].transform("mean"); fmean = d.groupby("film")[m].transform("mean")
    between_dir = ((dmean - gm) ** 2).sum() / tot
    between_film = ((fmean - dmean) ** 2).sum() / tot
    rows.append(dict(group=GROUP_OF[m], metric=m, films=n, H=H, p=p, eps2=eps2,
                     var_director=between_dir, var_film=between_film, var_shot=1 - between_dir - between_film))
res = pd.DataFrame(rows)
res["q_bh"] = multipletests(res["p"], method="fdr_bh")[1]
res = res.sort_values("eps2", ascending=False)

# director means on film means, for reading the direction of differences
fmeans = df.groupby(["style", "film"])[METRICS].mean().reset_index()
dir_means = fmeans.groupby("style")[METRICS].mean().T.round(3)

# leave-one-film-out classification
def lofo(cols):
    X = df[cols].values; y = df["style"].values; films = df["film"].values
    pred = np.empty(len(df), dtype=object)
    for f in np.unique(films):
        te = films == f
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
        clf.fit(X[~te], y[~te]); pred[te] = clf.predict(X[te])
    shot_acc = np.mean(pred == y)
    bal = np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])
    film_vote = pd.DataFrame(dict(film=films, y=y, p=pred)).groupby("film").agg(y=("y", "first"), p=("p", lambda s: s.mode()[0]))
    return dict(shot_acc=shot_acc, shot_balanced_acc=bal, film_acc=np.mean(film_vote.y == film_vote.p), films=len(film_vote)), \
        pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred"))

clf_rows, confs = [], {}
for name, cols in list(GROUPS.items()) + [("all", METRICS)]:
    r, cm = lofo(cols); r["features"] = name; r["n_features"] = len(cols); clf_rows.append(r); confs[name] = cm
clf = pd.DataFrame(clf_rows)[["features", "n_features", "shot_acc", "shot_balanced_acc", "film_acc", "films"]]

os.makedirs(OUT, exist_ok=True)
res.to_csv(os.path.join(OUT, "metric_discrimination.csv"), index=False)
dir_means.to_csv(os.path.join(OUT, "director_means_by_film.csv"))
clf.to_csv(os.path.join(OUT, "lofo_classification.csv"), index=False)
confs["all"].to_csv(os.path.join(OUT, "lofo_confusion_all.csv"))
pd.set_option("display.width", 200)
print(df.groupby("style").agg(shots=("film", "size"), films=("film", "nunique")).to_string()); print()
print(res.round(3).to_string(index=False)); print()
print(clf.round(3).to_string(index=False)); print()
print(confs["all"].to_string())
