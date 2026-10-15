"""English comparison table: each line of the trait table -> metric -> real dialogue shots vs prompts.
python trait_table.py REFDIR GENDIR"""
import sys, os, json, numpy as np, pandas as pd
R, G = sys.argv[1], sys.argv[2]
real = pd.DataFrame(json.load(open(os.path.join(R, "dialog_shots.json"))))
gen = pd.DataFrame(json.load(open(os.path.join(G, "gen_trait_metrics.json"))))
NAME = {"nolan": "Nolan", "burton": "Burton", "kubrick": "Kubrick", "wes_anderson": "Wes Anderson", "zhang_yimou": "Zhang Yimou", "spielberg": "Spielberg"}
MAP = [  # (director, trait as written in the table, metric, expected direction vs. a neutral film)
 ("nolan", "cool gray-blue", "warmth_b", "lower"), ("nolan", "low saturation", "saturation", "lower"),
 ("nolan", "IMAX large format", "aspect", "real only"),
 ("burton", "pale characters against dark backgrounds", "face_minus_bg", "higher"), ("burton", "high contrast", "contrast", "higher"),
 ("kubrick", "one-point perspective", "vp_converge", "higher"), ("kubrick", "one-point perspective (VP centered)", "vp_center", "higher"),
 ("kubrick", "centered symmetry", "symmetry", "higher"), ("kubrick", "slow dolly track", "push_in", "higher"),
 ("wes_anderson", "frontal eye-level view", "frontal_faces", "higher"), ("wes_anderson", "centered symmetry", "symmetry", "higher"),
 ("wes_anderson", "pastel colors", "pastel", "higher"), ("wes_anderson", "lateral tracking and pans", "lateral", "higher"),
 ("zhang_yimou", "large areas of single colors such as red and yellow", "red_yellow", "higher"),
 ("spielberg", "warm backlight", "warmth_b", "higher"), ("spielberg", "warm backlight, light shafts", "backlight", "higher"),
 ("spielberg", "slow push-in to the faces", "face_growth", "higher"), ("spielberg", "long take", "shot_seconds", "higher")]
NOT_MEAS = {"nolan": "real locations; realistic and weighty", "burton": "gothic; expressionist design; black-and-white stripes",
            "kubrick": "wide-angle lens; cold and detached", "wes_anderson": "meticulous sets",
            "zhang_yimou": "color symbolism; ritualized composition", "spielberg": "staging"}
allreal = real  # pooled real dialogue shots of all six directors = neutral reference
def g(model, prompt, style, k):
    v = gen[(gen.source == model) & (gen.prompt == prompt) & (gen["style"] == style)][k]
    return v.mean() if len(v) else np.nan
rows = []
for d, trait, k, exp in MAP:
    rv = real[real["style"] == d][k].dropna() if k in real else pd.Series(dtype=float)
    row = {"Director": NAME[d], "Trait (prompt wording)": trait, "Metric": k, "Expected": exp,
           "Real median": rv.median(), "Real IQR": f"{rv.quantile(.25):.2f} to {rv.quantile(.75):.2f}" if len(rv) else "",
           "Real n shots": len(rv), "All-director median": allreal[k].median() if k in allreal else np.nan}
    for m, tag in [("wan-3.0", "Wan"), ("veo-3.1-lite", "Veo")]:
        if k == "aspect": row.update({f"{tag} none": np.nan, f"{tag} name": np.nan, f"{tag} table": np.nan}); continue
        row[f"{tag} none"] = g(m, "none", "none", k); row[f"{tag} name"] = g(m, "name", d, k); row[f"{tag} table"] = g(m, "table", d, k)
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(G, "trait_comparison.csv"), index=False, float_format="%.3f")
num = [c for c in T.columns if T[c].dtype.kind == "f"]
md = T.copy(); md[num] = md[num].round(2)
open(os.path.join(G, "trait_comparison.md"), "w").write(
    "# Trait table vs. measurements (real dialogue shots vs. generated diner clips)\n\n" + md.to_markdown(index=False) +
    "\n\nNot measurable with current tools (human or VLM rating only): " + "; ".join(f"{NAME[k]}: {v}" for k, v in NOT_MEAS.items()) +
    ".\n\nGenerated clips are one 4 to 5 s shot, so shot length and camera moves are capped; generation is fixed at 16:9.\n")
print(md.drop(columns=["Expected"]).to_string(index=False))
