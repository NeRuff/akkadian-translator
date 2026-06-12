import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

model_path = './models/optimized/final'
print(f"Загрузка модели из {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def translate(text, beam_size=4):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
    outputs = model.generate(
        **inputs,
        num_beams=beam_size,
        max_length=128,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

df = pd.read_csv('data/train.csv')
eval_df = df.sample(min(50, len(df)), random_state=42)

references = []
hypotheses = []

print("Оценка качества на 50 случайных примерах...")
for _, row in eval_df.iterrows():
    pred = translate(row['transliteration'], beam_size=4)
    hypotheses.append(pred)
    references.append(row['translation'])

bleu = sacrebleu.corpus_bleu(hypotheses, [references])
chrf = sacrebleu.corpus_chrf(hypotheses, [references])

print(f"BLEU: {bleu.score:.2f}")
print(f"chrF++: {chrf.score:.2f}")

print("\nПримеры переводов:")
for i in range(min(3, len(eval_df))):
    print(f"\nAkkadian: {eval_df.iloc[i]['transliteration'][:80]}...")
    print(f"Reference: {eval_df.iloc[i]['translation'][:80]}...")
    print(f"Predicted: {hypotheses[i][:80]}...")
