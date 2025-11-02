from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_excel("data/esnli_selected.xlsx")

def translate_text(text):
    if pd.isna(text):
        return ""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a professional English-German translator for Natural Language Inference (NLI) data."
                                          "Translate each field into fluent, grammatically correct German "
                                          "while preserving meaning, logic, and tone. Keep numbers, entities, and structure unchanged."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

tqdm.pandas()
df["Sentence1_de"] = df["Sentence1"].progress_apply(translate_text)
df["Sentence2_de"] = df["Sentence2"].progress_apply(translate_text)
df["Explanation_1_de"] = df["Explanation_1"].progress_apply(translate_text)

df.to_excel("esnli_de_translated.xlsx", index=False)
