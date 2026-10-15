"""Pick real CMD clips per director: up to 4 live-action films, 2 clips each, dialogue-like first."""
import pandas as pd, sys
C = "/Users/apple/Desktop/EN.601.762_Multimodal_Fall2026/final_project/data/cmd/"
c = pd.read_csv(C + "clips_merged.csv").merge(pd.read_csv(C + "descriptions.csv")[["videoid", "description"]], on="videoid", how="left")
KW = r"talk|convers|dinner|restaurant|diner|cafe|café|bar|lunch|breakfast|tea|table|meet|discuss|tells|asks|confront|argu"
DROP = {"Home", "Fantastic Mr. Fox", "The Adventures of Tintin"}
DIRS = {"nolan": "Christopher Nolan", "burton": "Tim Burton", "kubrick": "Stanley Kubrick",
        "wes_anderson": "Wes Anderson", "zhang_yimou": "Zhang Yimou", "spielberg": "Steven Spielberg"}
rows = []
for key, dn in DIRS.items():
    s = c[c.director.fillna("").str.contains(dn) & ~c.title.isin(DROP)].copy()
    s["dlg"] = (s.description.fillna("") + " " + s.clip_name.fillna("")).str.contains(KW, case=False)
    films = s.groupby("title").dlg.sum().sort_values(ascending=False).index.tolist()
    picked = []
    for f in films[:4]:
        picked += s[s.title == f].sort_values("dlg", ascending=False).head(2).to_dict("records")
    for f in films[4:]:                       # top up when fewer than 4 films (Wes, Zhang)
        if len(picked) >= 8: break
        picked += s[s.title == f].head(1).to_dict("records")
    if len(picked) < 8:
        extra = s[~s.videoid.isin([p["videoid"] for p in picked])].sort_values("dlg", ascending=False)
        picked += extra.head(8 - len(picked)).to_dict("records")
    for p in picked: rows.append(dict(style=key, videoid=p["videoid"], title=p["title"], year=p["year"], clip=p["clip_name"]))
out = pd.DataFrame(rows); out.to_csv(sys.argv[1], index=False)
print(out.groupby("style").agg(n=("videoid", "size"), films=("title", "nunique")))
