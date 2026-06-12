import pandas as pd
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

os.environ["WANDB_DISABLED"] = "true"

df = pd.read_csv('data/train_augmented.csv' if os.path.exists('data/train_augmented.csv') else 'data/train.csv')
print(f"Загружено {len(df)} примеров")

train_df = df.sample(frac=0.9, random_state=42)
eval_df = df.drop(train_df.index)

train_dataset = Dataset.from_pandas(train_df[['transliteration', 'translation']].rename(
    columns={'transliteration': 'source', 'translation': 'target'}))
eval_dataset = Dataset.from_pandas(eval_df[['transliteration', 'translation']].rename(
    columns={'transliteration': 'source', 'translation': 'target'}))

model_name = "google/byt5-base"
print(f"Загрузка {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(examples):
    inputs = tokenizer(examples['source'], max_length=256, truncation=True, padding='max_length')
    targets = tokenizer(examples['target'], max_length=256, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']
    return inputs

train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir='./models/final',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    weight_decay=0.05,
    save_total_limit=2,
    num_train_epochs=30,
    predict_with_generate=True,
    generation_max_length=512,
    generation_num_beams=4,
    logging_dir='./logs',
    logging_steps=50,
    report_to='none',
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("🔥 Начинаю обучение...")
trainer.train()
model.save_pretrained('./models/final/final')
tokenizer.save_pretrained('./models/final/final')
print("✅ Модель сохранена в ./models/final/final")
