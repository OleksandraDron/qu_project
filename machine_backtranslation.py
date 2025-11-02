from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_excel("data/esnli_de_gpt4_1_mini.xlsx")

def back_translate_text(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a professional German-English translator for Natural Language Inference (NLI) data."
                                          "Translate each field into fluent, grammatically correct English "
                                          "while preserving meaning, logic, and tone. Keep numbers, entities, and structure unchanged."},
            {"role": "user", "content": text}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

tqdm.pandas(desc="Back-translating Sentence1")
df["Sentence1_back_en"] = df["Sentence1_de"].progress_apply(back_translate_text)
tqdm.pandas(desc="Back-translating Sentence2")
df["Sentence2_back_en"] = df["Sentence2_de"].progress_apply(back_translate_text)
tqdm.pandas(desc="Back-translating Explanation_1")
df["Explanation_1_back_en"] = df["Explanation_1_de"].progress_apply(back_translate_text)

df.to_excel("esnli_en_gpt4_1.xlsx", index=False)

