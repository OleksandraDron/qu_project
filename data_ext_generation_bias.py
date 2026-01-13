from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "data/esnli_de_corrected_updated_un.xlsx"
OUTPUT_FILE = "data/esnli_de_corrected_updated_ni.xlsx"

df = pd.read_excel(INPUT_FILE)

def build_evaluation_prompt(premise: str, hypothesis: str, explanation: str) -> str:
    return f"""
Du bist ein Assistent, der Erklärungen für Premise-Hypothesis-Paare bewertet. 
Deine Aufgabe ist es, die gegebene Explanation ausschließlich nach dem folgenden Kriterium zu bewerten:

New Information – Die Erklärung führt Informationen ein, die nicht aus der Prämisse, der Hypothese oder allgemeinem Weltwissen ableitbar sind.

Bewertung:
0 = nein  
1 = ja

Gegeben sind drei Felder:
Premise: "{premise}"
Hypothesis: "{hypothesis}"
Explanation: "{explanation}"

Beachte diese Beispiele:

Beispiel 1 :
Premise: Eine Katze schläft auf dem Sofa.
Hypothesis: Die Katze liegt auf einem Möbelstück.
Explanation: Ein Sofa ist ein Möbelstück, und Schlafen impliziert Liegen.
Bewertung: "New Information":0

Beispiel 2 :
Premise: Ein Mann joggt im Park.
Hypothesis: Der Mann trainiert im Freien.
Explanation: Der Mann trägt einen blauen Trainingsanzug.
Bewertung: "New Information":1

Beispiel 3 :
Premise: Ein Kind isst ein Eis.
Hypothesis: Das Kind genießt eine Süßigkeit.
Explanation: Eis ist eine gefrorene Süßigkeit.
Bewertung: "New Information":0

Beispiel 4 :
Premise: Eine Frau telefoniert am Flughafen.
Hypothesis: Die Frau spricht mit jemandem.
Explanation: Sie wartet auf einen Flug nach Berlin.
Bewertung: "New Information":1

Beispiel 5 :
Premise: Ein Junge spielt Gitarre.
Hypothesis: Der Junge macht Musik.
Explanation: Der Junge spielt Rockmusik und singt lautstark dazu.
Bewertung: "New Information":1

Beispiel 6 :
Premise: Eine Frau liest ein Buch.
Hypothesis: Die Frau liest.
Explanation: Die Frau hält ein Buch in der Hand und liest darin.
Bewertung: "New Information":0


Bewerte nun die folgende Explanation nur im Hinblick auf New Information.

Antworte nur mit:
New Information: 0/1

""".strip()


# ======================
# CONFIG: choose ONE metric per run
# ======================
TARGET_METRIC = "New Information"
# options:
# "Well-Written"
# "Related"
# "Factual"
# "New Information"
# "Unnecessary Information"

METRIC_PATTERNS = {
    "Well-Written": r"Well-written:\s*([01])",
    "Related": r"Related:\s*([01])",
    "Factual": r"Factual:\s*([01])",
    "New Information": r"New Information:\s*([01])",
    "Unnecessary Information": r"Unnecessary Information:\s*([01])"
}
# ======================
# Parse single score
# ======================
def parse_single_score(text: str, metric: str):
    pattern = METRIC_PATTERNS[metric]
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


# ======================
# Evaluation function
# ======================
def evaluate_row(row):
    if (
        pd.isna(row["Sentence1_de"])
        or pd.isna(row["Sentence2_de"])
        or pd.isna(row["Explanation_1_de"])
    ):
        return row

    prompt = build_evaluation_prompt(
        premise=row["Sentence1_de"],
        hypothesis=row["Sentence2_de"],
        explanation=row["Explanation_1_de"]
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Du bewertest Erklärungen strikt nach dem angegebenen Kriterium."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=80
    )

    content = response.choices[0].message.content
    score = parse_single_score(content, TARGET_METRIC)

    row[TARGET_METRIC] = score
    return row


# ======================
# Run
# ======================
tqdm.pandas()

df = df.progress_apply(evaluate_row, axis=1)

df.to_excel(OUTPUT_FILE, index=False)

print(f"✅ Evaluation for metric '{TARGET_METRIC}' completed and saved to:", OUTPUT_FILE)
