"""Where do generated clips fall in the real-film distribution? Real = CMD shots (dialogue shots by default).
For each metric: real median and interquartile range, generated median per model, and the percentile of the generated
median inside the real shot distribution. Encoding (bits per pixel) is reported for both sides.

python compare_gen.py REAL_EXTRACT GEN_EXTRACT OUTCSV [--all-shots]"""
import sys, os, json, glob, argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("real"); ap.add_argument("gen"); ap.add_argument("out"); ap.add_argument("--all-shots", action="store_true")
a = ap.parse_args()


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "shots", "*.json")):
        j = json.load(open(f))
        for s in j["shots"]:
            s.update({f"clip_{k}": v for k, v in j["clip"].items()}); rows.append(s)
    return pd.DataFrame(rows)


R, G = load(a.real), load(a.gen)
R = R[~R["black"].astype(bool)]
if not a.all_shots: R = R[R["dialog"].astype(bool)]
metrics = [c for c in R.columns if c[:2] in ("CM", "CA", "CO", "ST", "LI", "CL", "GR", "TX") and pd.api.types.is_numeric_dtype(R[c])
           and c in G.columns and not c.startswith("CA11")]
rows = []
for m in metrics:
    r = R[m].replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 50: continue
    row = dict(metric=m, real_n=len(r), real_p25=r.quantile(.25), real_median=r.median(), real_p75=r.quantile(.75))
    for model, g in G.groupby("title"):
        v = g[m].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 3: continue
        md = v.median(); row[f"{model}_median"] = md; row[f"{model}_pct_in_real"] = float((r < md).mean() * 100)
    rows.append(row)
out = pd.DataFrame(rows)
enc = pd.DataFrame([dict(side="real", bpp_median=R["clip_bits_per_pixel"].median(), height=R["clip_src_height"].median())] +
                   [dict(side=m, bpp_median=g["clip_bits_per_pixel"].median(), height=g["clip_src_height"].median()) for m, g in G.groupby("title")])
out.to_csv(a.out, index=False); enc.to_csv(a.out.replace(".csv", "_encoding.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(enc.round(4).to_string(index=False)); print()
print(out.round(3).to_string(index=False))
