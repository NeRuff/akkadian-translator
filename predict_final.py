import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from tqdm import tqdm

model_path = './models/final/final'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def translate(text, beam_size=8):
    inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True)
    outputs = model.generate(
        **inputs,
        num_beams=beam_size,
        max_length=512,
        early_stopping=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=0.6,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result and result[-1] not in '.!?':
        result += '.'
    return result

test = pd.read_csv('data/test.csv')
print(f"Тестовых примеров: {len(test)}")

predictions = []
for text in tqdm(test['transliteration'], desc="Перевод"):
    predictions.append(translate(text))

submission = pd.DataFrame({'id': test['id'], 'translation': predictions})
submission.to_csv('data/submission_final.csv', index=False)

eval_df = pd.read_csv('data/train.csv').sample(100, random_state=42)
refs = []
hyps = []
for _, row in eval_df.iterrows():
    hyps.append(translate(row['transliteration']))
    refs.append(row['translation'])
chrf = sacrebleu.corpus_chrf(hyps, [refs])
print(f"chrF++ на валидации: {chrf.score:.2f}")
