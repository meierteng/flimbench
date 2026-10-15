"""Ask LLMs to state a director's style for the diner scene as concrete parameters (what the model *thinks* the style is).
python verbalize.py OUT.json"""
import json, os, sys, re, requests, concurrent.futures as cf
LLMS = ["qwen/qwen3.8-max-0902", "google/gemini-3.8-flash", "openai/gpt-5.4"]
DIRS = {"none": None, "nolan": "Christopher Nolan", "burton": "Tim Burton", "kubrick": "Stanley Kubrick",
        "wes_anderson": "Wes Anderson", "zhang_yimou": "Zhang Yimou", "spielberg": "Steven Spielberg"}
SCENE = "Two people sit across from each other at a diner table, talking quietly over coffee."
SCHEMA = """Return only JSON with these keys:
"shot_scale": one of ["extreme wide","wide","full","medium","medium close-up","close-up","extreme close-up"],
"camera_angle": one of ["high","eye level","low","overhead","dutch"],
"camera_position": one of ["frontal","profile/side","three-quarter","over-the-shoulder"],
"camera_movement": one of ["static","pan","tilt","dolly/track in","dolly/track out","lateral track","handheld","crane","zoom"],
"lens": one of ["wide","normal","telephoto"],
"composition": one of ["centered symmetrical","rule of thirds","off-center","deep one-point perspective"],
"number_of_shots_in_5_seconds": integer,
"average_shot_length_seconds_in_this_directors_films": number,
"brightness": 1-5 (3 = typical contemporary Hollywood film, 1 much darker, 5 much brighter),
"saturation": 1-5 (same anchor),
"warmth": 1-5 (1 much cooler/bluer, 5 much warmer/yellower),
"contrast": 1-5,
"palette": short phrase,
"lighting": short phrase,
"production_design_and_costume": short phrase,
"video_prompt": the exact text-to-video prompt (under 80 words) you would give a video model to get this shot"""
def ask(llm, key):
    who = DIRS[key]
    task = (f"Scene: {SCENE}\nDescribe how this single 5-second shot would look " +
            (f"if directed by {who}, faithful to how {who} actually shoots such scenes." if who else "in a typical contemporary film.") +
            "\n" + SCHEMA)
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=180,
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      json={"model": llm, "messages": [{"role": "user", "content": task}], "temperature": 0,
                            "response_format": {"type": "json_object"}})
    j = r.json(); txt = j["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    return dict(llm=llm, style=key, cost=(j.get("usage") or {}).get("cost"), **json.loads(m.group(0)))
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lambda t: ask(*t), [(l, k) for l in LLMS for k in DIRS]))
json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print("cost", round(sum(float(r["cost"] or 0) for r in res), 4))
import pandas as pd
df = pd.DataFrame(res)
print(df[["llm", "style", "shot_scale", "camera_position", "camera_movement", "composition", "number_of_shots_in_5_seconds",
          "average_shot_length_seconds_in_this_directors_films", "brightness", "saturation", "warmth", "contrast"]].to_string())
