from openai import OpenAI
import pandas as pd
from tqdm import tqdm

client = OpenAI(api_key="OPENAI_API_KEY")

df = pd.read_csv("esnli_sample.csv")

def translate_text(text):
    if pd.isna(text):
        return ""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional translator. Translate the text from English to German accurately and naturally."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

tqdm.pandas()
df["Sentence1_de"] = df["Sentence1"].progress_apply(translate_text)
df["Sentence2_de"] = df["Sentence2"].progress_apply(translate_text)
df["Explanation_1_de"] = df["Explanation_1"].progress_apply(translate_text)

df.to_csv("esnli_de_translated.csv", index=False)
