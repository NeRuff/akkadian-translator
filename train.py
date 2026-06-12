import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import os

os.environ["WANDB_DISABLED"] = "true"

print("Загрузка данных...")
df = pd.read_csv('data/train_norm.csv' if os.path.exists('data/train_norm.csv') else 'data/train.csv')

if 'transliteration_norm' in df.columns:
    source_col = 'transliteration_norm'
    target_col = 'translation_norm'
else:
    source_col = 'transliteration'
    target_col = 'translation'

train_df = df.sample(frac=0.9, random_state=42)
eval_df = df.drop(train_df.index)
print(f"Train: {len(train_df)}, Dev: {len(eval_df)}")

train_dataset = Dataset.from_pandas(train_df[[source_col, target_col]].rename(columns={source_col: 'source', target_col: 'target'}))
eval_dataset = Dataset.from_pandas(eval_df[[source_col, target_col]].rename(columns={source_col: 'source', target_col: 'target'}))

model_name = "google/byt5-small"
print(f"Загрузка модели {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(examples):
    inputs = tokenizer(examples['source'], max_length=128, truncation=True, padding='max_length')
    targets = tokenizer(examples['target'], max_length=128, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']
    return inputs

print("Токенизация...")
train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset = eval_dataset.map(preprocess, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir='./models/baseline',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=50,
    report_to='none',
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("Начинаю обучение...")
trainer.train()
print("Обучение завершено!")

model.save_pretrained('./models/baseline/final')
tokenizer.save_pretrained('./models/baseline/final')
print("Модель сохранена в ./models/baseline/final")
