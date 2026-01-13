from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_excel("data/esnli_de_corrected.xlsx")

# def build_prompt(premise: str, hypothesis: str, label: str) -> str:
#     return (
#         "Gegeben ist eine Aufgabe zur natürlichen Sprachinferenz (NLI):\n"
#         f'Premise: "{premise}"\n'
#         f'Hypothesis: "{hypothesis}"\n'
#         f"Label: {label}.\n\n"
#         "Bitte gib eine kurze (1–3 Sätze), klar formulierte und logisch konsistente Erklärung auf Deutsch, "
#         "die beschreibt, warum dieses Label korrekt ist. "
#         "Vermeide es, neue Informationen einzuführen, die nicht aus der Prämisse ableitbar sind."
#     )


# def build_prompt(premise: str, hypothesis: str) -> str:
#     return (
#         "Gegeben sind zwei Sätze:\n"
#         f'Premise: "{premise}"\n'
#         f'Hypothesis: "{hypothesis}"\n\n'
#         "Schreibe eine sehr kurze Erklärung auf Deutsch (max. 1–2 Sätze), "
#         "die den entscheidenden inhaltlichen Unterschied oder Zusammenhang benennt.\n\n"
#         "Wichtige Vorgaben:\n"
#         "- Verwende einfache, konkrete Aussagen über die beschriebene Situation.\n"
#         "- Erwähne weder Premise, Hypothese noch das Label.\n"
#         "- Wiederhole die Sätze nicht und paraphrasiere sie nicht vollständig."
#     )

def build_prompt(premise: str, hypothesis: str) -> str:
    return (
        "Gegeben sind zwei Sätze:\n"
        f'Premise: "{premise}"\n'
        f'Hypothesis: "{hypothesis}"\n\n'
        "Schreibe eine sehr kurze Erklärung auf Deutsch (1–2 kurze Sätze), "
        "die die entscheidenden, konkreten Fakten nennt, die den Zusammenhang erklären.\n\n"
        "Wichtige Vorgaben:\n"
        "- Verwende konkrete Tatsachenbehauptungen (z.B. \"X ist nicht Y\", \"entweder X oder Y\").\n"
        "- Verwende einfache, konkrete Aussagen über die beschriebene Situation.\n"
        "- Erwähne weder Premise, Hypothese noch das Label.\n"
        "- Die Erklärung darf aus sehr einfachen oder verkürzten Sätzen bestehen."
        "- Wiederhole die Sätze nicht und paraphrasiere sie nicht vollständig."
    )


def generate_explanation_de(row):
    if pd.isna(row["Sentence1_de"]) or pd.isna(row["Sentence2_de"]) or pd.isna(row["gold_label"]):
        return ""

    prompt = build_prompt(
        premise=row["Sentence1_de"],
        hypothesis=row["Sentence2_de"]
        # label=row["gold_label"]
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein linguistisch präziser Assistent für "
                    "Natural Language Inference. Die Erklärung muss "
                    "kurz, sachlich und logisch korrekt sein."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )

    return response.choices[0].message.content.strip()

tqdm.pandas()
df["Explanation_de_generated"] = df.progress_apply(generate_explanation_de, axis=1)

# Save result
df.to_excel("data/esnli_de_generated_ger_explanations_gpt41mini.xlsx", index=False)
