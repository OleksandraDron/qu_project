import pandas as pd
import sacrebleu

df = pd.read_excel("data/esnli_en_gpt4_1.xlsx")

# for col in ["Sentence1","Sentence2","Explanation_1",
#             "Sentence1_back_en","Sentence2_back_en","Explanation_1_back_en"]:
#     if col in df.columns:
#         df[col] = df[col].fillna("")

def pairs(orig_col, back_col):
    src = df[orig_col].astype(str).tolist()
    bt  = df[back_col].astype(str).tolist()
    return src, bt

src_s1, bt_s1 = pairs("Sentence1", "Sentence1_back_en")
src_s2, bt_s2 = pairs("Sentence2", "Sentence2_back_en")
src_ex, bt_ex = pairs("Explanation_1", "Explanation_1_back_en")

src_all = src_s1 + src_s2 + src_ex
bt_all  = bt_s1  + bt_s2  + bt_ex

def bleu_chrf_corpus(sys, ref):
    bleu = sacrebleu.corpus_bleu(sys, [ref]).score
    chrf = sacrebleu.corpus_chrf(sys, [ref]).score
    return bleu, chrf

bleu_all, chrf_all = bleu_chrf_corpus(bt_all, src_all)
print(f"[ALL]  BLEU: {bleu_all:.2f} | chrF: {chrf_all:.2f}")

for name, sys, ref in [
    ("Sentence1", bt_s1, src_s1),
    ("Sentence2", bt_s2, src_s2),
    ("Explanation_1", bt_ex, src_ex),
]:
    b, c = bleu_chrf_corpus(sys, ref)
    print(f"[{name}] BLEU: {b:.2f} | chrF: {c:.2f}")