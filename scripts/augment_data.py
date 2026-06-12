import pandas as pd
import random
import re

def swap_words(text):
    words = text.split()
    if len(words) < 4:
        return text
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def drop_random(text):
    words = text.split()
    if len(words) < 3:
        return text
    drop_idx = random.randint(0, len(words)-1)
    return ' '.join(words[:drop_idx] + words[drop_idx+1:])

df = pd.read_csv('data/train.csv')
print(f"Оригинальных примеров: {len(df)}")

augmented = []
for _, row in df.iterrows():
    augmented.append({'transliteration': row['transliteration'], 'translation': row['translation']})
    
    if random.random() > 0.5:
        augmented.append({
            'transliteration': swap_words(row['transliteration']),
            'translation': row['translation']
        })
    
    if random.random() > 0.7:
        augmented.append({
            'transliteration': drop_random(row['transliteration']),
            'translation': row['translation']
        })

aug_df = pd.DataFrame(augmented)
aug_df.to_csv('data/train_augmented.csv', index=False)
print(f"Всего примеров после аугментации: {len(aug_df)}")
