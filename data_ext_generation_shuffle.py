from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import re

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "data/esnli_de_corrected_updated.xlsx"
OUTPUT_FILE = "data/esnli_de_corrected_updated_shuffle.xlsx"

df = pd.read_excel(INPUT_FILE)


def build_evaluation_prompt(premise: str, hypothesis: str, explanation: str) -> str:
    return f"""
Du bist ein Assistent, der Erklärungen für Premise-Hypothesis-Paare bewertet. 
Deine Aufgabe ist es, jede Explanation anhand von fünf Kriterien zu bewerten: 0 = nein, 1 = ja.

Gegeben sind drei Felder:
Premise: "{premise}"
Hypothesis: "{hypothesis}"
Explanation: "{explanation}"

Bevor du die Bewertung durchführst, beachte diese Beispiele:

Beispiel 1 :
Premise: Eine Katze schläft auf dem Sofa.
Hypothesis: Die Katze liegt auf einem Möbelstück.
Explanation: Ein Sofa ist ein Möbelstück, und Schlafen impliziert Liegen.
Bewertung: "Factual":1,"Related":1,"New Information":0,"Well-written":1,"Unnecessary Information":0

Beispiel 2 :
Premise: Ein Mann joggt im Park.
Hypothesis: Der Mann trainiert im Freien.
Explanation: Der Mann trägt einen blauen Trainingsanzug.
Bewertung: "Factual":0,"Related":1,"New Information":1,"Well-written":1,"Unnecessary Information":0

Beispiel 3 :
Premise: Ein Kind isst ein Eis.
Hypothesis: Das Kind genießt eine Süßigkeit.
Explanation: Eis ist eine gefrorene Süßigkeit.
Bewertung: "Factual":1,"Related":1,"New Information":0,"Well-written":1,"Unnecessary Information":0

Beispiel 4 :
Premise: Eine Frau telefoniert am Flughafen.
Hypothesis: Die Frau spricht mit jemandem.
Explanation: Sie wartet auf einen Flug nach Berlin.
Bewertung: "Factual":0,"Related":0,"New Information":1,"Well-written":1,"Unnecessary Information":0

Beispiel 5 :
Premise: Ein Junge spielt Gitarre.
Hypothesis: Der Junge macht Musik.
Explanation: Der Junge spielt Rockmusik und singt lautstark dazu.
Bewertung: "Factual":0,"Related":1,"New Information":1,"Well-written":1,"Unnecessary Information":1

Beispiel 6 :
Premise: Eine Frau liest ein Buch.
Hypothesis: Die Frau liest.
Explanation: Die Frau hält ein Buch in der Hand und liest darin.
Bewertung: "Factual":1,"Related":1,"New Information":0,"Well-written":1,"Unnecessary Information":1

Nun bewerte die folgende Explanation anhand der Kriterien:
Factual – basiert nur auf den gegebenen Sätzen oder allgemeingültigen Definitionen,
Related – bezieht sich direkt auf den Zusammenhang zwischen Prämisse und Hypothese,
New Information – führt Informationen ein, die nicht aus den Sätzen ableitbar sind,
Well-written – klar und verständlich,
Unnecessary Information – enthält Details, die für die Begründung der Hypothese irrelevant sind

Antworte **nur** im folgenden Format:

Factual: 0/1  
Related: 0/1  
New Information: 0/1  
Well-written: 0/1 
Unnecessary Information: 0/1
""".strip()
# ======================
# Parsing helper
# ======================
def parse_scores(text: str) -> dict:
    scores = {
        "Factual": None,
        "Related": None,
        "New Information": None,
        "Well-Written": None,
        "Unnecessary Information": None
    }

    patterns = {
        "Factual": r"Factual:\s*([01])",
        "Related": r"Related:\s*([01])",
        "New Information": r"New Information:\s*([01])",
        "Well-Written": r"Well-written:\s*([01])",
        "Unnecessary Information": r"Unnecessary Information:\s*([01])"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            scores[key] = int(match.group(1))

    return scores

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
                "content": "Du bewertest Erklärungen strikt nach vorgegebenen Kriterien."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=150
    )

    content = response.choices[0].message.content
    scores = parse_scores(content)

    for col, value in scores.items():
        row[col] = value

    return row

# ======================
# Run
# ======================
tqdm.pandas()

df = df.progress_apply(evaluate_row, axis=1)

df.to_excel(OUTPUT_FILE, index=False)

print("✅ Evaluation completed and saved to:", OUTPUT_FILE)
