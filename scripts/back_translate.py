import pandas as pd
from transformers import pipeline

print("Загрузка модели для обратного перевода...")
reverse_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")

df = pd.read_csv('data/train.csv')
print(f"Оригинальных примеров: {len(df)}")

synthetic = []
for _, row in df.iterrows():
    back = reverse_model(row['translation'], max_length=128)
    synthetic.append({
        'transliteration': back[0]['translation_text'],
        'translation': row['translation']
    })

synthetic_df = pd.DataFrame(synthetic)
combined = pd.concat([df[['transliteration', 'translation']], synthetic_df])
combined.to_csv('data/train_backtranslated.csv', index=False)
print(f"Всего примеров после back-translation: {len(combined)}")
