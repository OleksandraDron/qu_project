from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json

# --- Setup ---
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Input file (з already generated explanations)
df = pd.read_excel("data/esnli_de_exp_gpt41mini_100_2.xlsx")

# --- Evaluation prompt ---
def build_evaluation_prompt(premise: str, hypothesis: str, explanation: str) -> str:
    return f"""
Gegeben sind drei Felder:
Premise: "{premise}"
Hypothesis: "{hypothesis}"
Explanation: "{explanation}"

Bewerte die Explanation anhand dieser Kriterien (0 = nein, 1 = ja):
Well-written – klar und verständlich
Related – bezieht sich direkt auf die Sätze
Factual – enthält überprüfbare, konkrete Fakten
Contains new information – enthält zusätzliche Infos, die nicht in den Sätzen stehen
Contains unnecessary information – enthält irrelevante Infos

Gib die Ausgabe nur als kompakten JSON-Text zurück:
{{"Well-written":1,"Related":1,"Factual":1,"Contains new information":0,"Contains unnecessary information":0}}
""".strip()

# --- Single-row evaluation ---
def evaluate_explanation(row):
    if (
        pd.isna(row["Sentence1_de"])
        or pd.isna(row["Sentence2_de"])
        or pd.isna(row["Explanation_de_generated"])
    ):
        return None

    prompt = build_evaluation_prompt(
        premise=row["Sentence1_de"],
        hypothesis=row["Sentence2_de"],
        explanation=row["Explanation_de_generated"],
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein strenger, konsistenter Evaluator für "
                    "NLI-Erklärungen. Antworte ausschließlich mit validem JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_output": content}

# --- Run evaluation ---
tqdm.pandas()
df["evaluation"] = df.progress_apply(evaluate_explanation, axis=1)

# --- Save as JSON ---
output = []

for _, row in df.iterrows():
    if row["evaluation"] is None:
        continue

    output.append({
        "premise": row["Sentence1_de"],
        "hypothesis": row["Sentence2_de"],
        "explanation": row["Explanation_de_generated"],
        "evaluation": row["evaluation"],
    })

with open("data/esnli_de_explanation_evaluation_100.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ Evaluation saved to data/esnli_de_explanation_evaluation_100.json")
