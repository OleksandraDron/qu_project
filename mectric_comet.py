import os
os.environ["HF_HUB_USE_SYMLINKS_DEFAULT"] = "0"
import pandas as pd
from comet import download_model, load_from_checkpoint

df = pd.read_excel("data/esnli_de_gpt4_1.xlsx")

parts = []
if {"Sentence1","Sentence1_de"}.issubset(df.columns):
    parts.append(pd.DataFrame({"field":"Sentence1","src":df["Sentence1"],"mt":df["Sentence1_de"]}))
if {"Sentence2","Sentence2_de"}.issubset(df.columns):
    parts.append(pd.DataFrame({"field":"Sentence2","src":df["Sentence2"],"mt":df["Sentence2_de"]}))
if {"Explanation_1","Explanation_1_de"}.issubset(df.columns):
    parts.append(pd.DataFrame({"field":"Explanation_1","src":df["Explanation_1"],"mt":df["Explanation_1_de"]}))

df_long = pd.concat(parts, ignore_index=True).fillna("")
ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(ckpt)

data = [{"src": s, "mt": m} for s, m in zip(df_long["src"], df_long["mt"])]
out = model.predict(data, batch_size=32, gpus=0)

def parse_comet_output(o):
    if isinstance(o, dict):
        system = o.get("system_score")
        seg = o.get("segments_scores") or o.get("scores") or []
        return system, seg
    if isinstance(o, list):
        seg = o
        system = sum(seg)/len(seg) if seg else None
        return system, seg
    return None, []

system_score, seg_scores = parse_comet_output(out)

# save to csv details
# if seg_scores:
#     df_long = df_long.copy()
#     df_long["COMET_QE"] = seg_scores
#     df_long.rename(columns={"src":"original_en","mt":"translation_de"}).to_csv(
#         "comet_qe_segment_scores.csv", index=False, encoding="utf-8"
#     )

if seg_scores:
    for name, group in df_long.groupby("field"):
        field_score = group["COMET_QE"].mean()
        print(f"[{name}] COMET-QE: {field_score:.4f}")