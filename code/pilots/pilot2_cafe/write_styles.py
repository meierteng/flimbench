"""GPT-5.4 writes a four-system style description per director (no name, no change to characters or plot)."""
import json, os, requests
CONTENT = ("Two people sit across from each other at a cafe table, talking quietly over coffee. "
           "Then they stand up and walk out of the cafe together.")
DIRS = {"nolan": "Christopher Nolan", "burton": "Tim Burton", "kubrick": "Stanley Kubrick",
        "wes_anderson": "Wes Anderson", "zhang_yimou": "Zhang Yimou", "spielberg": "Steven Spielberg"}
ASK = """A text-to-video model will generate an 8-second clip of this action: "{c}"
Describe how {d} would film and finish this clip, faithful to how {d} actually works, as concrete visual parameters in four systems:
1. Mise-en-scene: set and costume colors, lighting (quality, direction, key), staging of the two people.
2. Cinematography: shot scale, camera angle and height, composition, lens, camera movement.
3. Image rendering and finishing: color grading, contrast and tonality, texture or grain, motion rendering.
4. Editing: number of shots in 8 seconds, where the cuts fall, transitions.
Rules: do not name the director or any film; do not change who the characters are or what they do; no costume ideas that change their identity.
Return only JSON: {{"mise_en_scene": "...", "cinematography": "...", "rendering": "...", "editing": "...", "prompt_suffix": "one comma-separated phrase under 70 words combining all four systems, to append to the action sentence"}}"""
H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
out, cost = {}, 0.0
for k, d in DIRS.items():
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=H, timeout=180, json={
        "model": "openai/gpt-5.4", "temperature": 0, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": ASK.format(c=CONTENT, d=d)}]}).json()
    out[k] = json.loads(r["choices"][0]["message"]["content"]); cost += float((r.get("usage") or {}).get("cost") or 0)
    print(k, "|", out[k]["prompt_suffix"])
json.dump({"content": CONTENT, "writer": "openai/gpt-5.4", "styles": out}, open("styles_gpt54.json", "w"), indent=1)
print("cost", round(cost, 4))
