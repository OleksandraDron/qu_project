import pandas as pd
import sacrebleu

df = pd.read_excel("data/esnli_en_gpt4_1_mini_200.xlsx")

def pairs(orig_col, back_col):
    src = df[orig_col].astype(str).tolist()
    bt  = df[back_col].astype(str).tolist()
    return src, bt

src_s1, bt_s1 = pairs("Sentence1", "Sentence1_back_en")
src_s2, bt_s2 = pairs("Sentence2", "Sentence2_back_en")

src_all = src_s1 + src_s2
bt_all  = bt_s1  + bt_s2

def bleu_chrf_corpus(sys, ref):
    bleu = sacrebleu.corpus_bleu(sys, [ref]).score
    chrf = sacrebleu.corpus_chrf(sys, [ref]).score
    return bleu, chrf

bleu_all, chrf_all = bleu_chrf_corpus(bt_all, src_all)
print(f"[ALL]  BLEU: {bleu_all:.4f} | chrF: {chrf_all:.4f}")

for name, sys, ref in [
    ("Sentence1", bt_s1, src_s1),
    ("Sentence2", bt_s2, src_s2),
]:
    b, c = bleu_chrf_corpus(sys, ref)
    print(f"[{name}] BLEU: {b:.4f} | chrF: {c:.4f}")