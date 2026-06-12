import re
import pandas as pd

def normalize_akkadian(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\[\.\.\.\]|⸢\.\.\.⸣|\<\.\.\.\>|\{\.\.\.\}', '', text)
    text = re.sub(r'([a-z])₂', r'\1', text)
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'₅|₄|₃|₂|₁', '', text)
    text = ' '.join(text.split())
    return text

def normalize_english(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\[.*?\]', '', text)
    text = ' '.join(text.split())
    return text

if __name__ == "__main__":
    df = pd.read_csv('data/train.csv')
    df['transliteration_norm'] = df['transliteration'].apply(normalize_akkadian)
    df['translation_norm'] = df['translation'].apply(normalize_english)
    df.to_csv('data/train_norm.csv', index=False)
    print(f"Сохранено {len(df)} строк в data/train_norm.csv")
    print(f"\nОригинал: {df['transliteration'].iloc[0][:100]}...")
    print(f"Нормализован: {df['transliteration_norm'].iloc[0][:100]}...")
