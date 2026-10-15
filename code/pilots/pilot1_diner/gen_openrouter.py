"""Same content and seed; only the director phrase changes. OpenRouter video API.
python gen_openrouter.py --out DIR [--models m1,m2] [--styles a,b] [--seeds 1,2] [--limit N]"""
import argparse, json, os, time, itertools, requests
API = "https://openrouter.ai/api/v1/videos"
ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_api.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--models", default=""); ap.add_argument("--styles", default=""); ap.add_argument("--seeds", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--style_key", default="styles")  # "styles_traits" = trait descriptions without the name
a = ap.parse_args()
P = json.load(open(a.prompts)); os.makedirs(a.out, exist_ok=True)
H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
pick = lambda s, d: s.split(",") if s else list(d)
models, styles = pick(a.models, P["models"]), pick(a.styles, P[a.style_key])
tag = {"styles": "", "styles_traits": "traits_", "styles_table": "table_"}[a.style_key]
seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else P["seeds"]
(cname, content), = P["content"].items()
jobs = list(itertools.product(models, styles, seeds))[: a.limit or None]

pending = {}
for m, s, sd in jobs:
    name = f"{m.split('/')[1]}__{cname}__{tag}{s}__s{sd}"
    if os.path.exists(os.path.join(a.out, name + ".mp4")): continue
    body = {"model": m, "prompt": content + P[a.style_key][s], "seed": sd, **P["models"][m]}
    r = requests.post(API, headers=H, json=body, timeout=60)
    if r.status_code != 202 and r.status_code != 200:
        print("SUBMIT FAIL", name, r.status_code, r.text[:300], flush=True); continue
    pending[name] = (r.json()["id"], body, time.time()); print("submitted", name, flush=True)

total = 0.0
while pending:
    time.sleep(20)
    for name, (jid, body, t0) in list(pending.items()):
        j = requests.get(f"{API}/{jid}", headers=H, timeout=60).json()
        st = j.get("status")
        if st == "completed":
            v = requests.get(f"{API}/{jid}/content?index=0", headers=H, timeout=300)
            open(os.path.join(a.out, name + ".mp4"), "wb").write(v.content)
            cost = (j.get("usage") or {}).get("cost"); total += float(cost or 0)
            json.dump({**body, "job": jid, "cost": cost, "sec": round(time.time() - t0)},
                      open(os.path.join(a.out, name + ".json"), "w"), ensure_ascii=False, indent=1)
            print("done", name, "cost", cost, flush=True); del pending[name]
        elif st in ("failed", "cancelled", "expired"):
            print("FAILED", name, json.dumps(j.get("error"))[:300], flush=True)
            json.dump({**body, "job": jid, "status": st, "error": j.get("error")},
                      open(os.path.join(a.out, name + ".failed.json"), "w"), indent=1); del pending[name]
print("total cost", round(total, 3))
