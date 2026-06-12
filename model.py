import typer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Iterator, Union
from loguru import logger
import sys
from pathlib import Path

logger.add("logs/model.log", rotation="10 MB")
logger.add(sys.stdout, level="INFO")

app = typer.Typer()

class My_Translator_Model:
    def __init__(self, model_path: str = "./models/optimized/final"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            logger.info(f"Модель загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

    def train(self, dataset_path: str) -> None:
        logger.info(f"Обучение на {dataset_path}")
        import subprocess
        result = subprocess.run(["python", "train_optimized.py"], capture_output=True, text=True)
        logger.info(result.stdout)

    def predict(self, text: str, stream: bool = True) -> Union[Iterator[str], str]:
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
        outputs = self.model.generate(**inputs, num_beams=4, max_length=128)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if stream:
            for word in translation.split():
                yield word + " "
        else:
            return translation

    def predict_file(self, dataset_path: str) -> None:
        logger.info(f"Предсказание для {dataset_path}")
        if self.model is None:
            self.load_model()

        df = pd.read_csv(dataset_path)
        predictions = []

        for _, row in df.iterrows():
            source = row.get('transliteration', row.get('akkadian', row.get('source', '')))
            if not source:
                continue
            inputs = self.tokenizer(source, return_tensors='pt', max_length=128, truncation=True)
            outputs = self.model.generate(**inputs, num_beams=4, max_length=128)
            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)

        result_df = pd.DataFrame({
            'id': df.get('id', df.get('oare_id', range(len(predictions)))),
            'translation': predictions
        })

        Path("data").mkdir(exist_ok=True)
        result_df.to_csv("./data/results.csv", index=False)
        logger.success(f"Сохранено в ./data/results.csv")

@app.command()
def train(
    dataset: str = typer.Option("data/train.csv", "--dataset", "-d", help="Путь к тренировочным данным")
):
    model = My_Translator_Model()
    model.train(dataset)

@app.command()
def predict(
    text: str = typer.Argument(..., help="Текст для перевода")
):
    model = My_Translator_Model()
    for token in model.predict(text, stream=True):
        print(token, end="", flush=True)
    print()

@app.command()
def predict_file(
    dataset: str = typer.Argument(..., help="Путь к тестовому файлу")
):
    model = My_Translator_Model()
    model.predict_file(dataset)

if __name__ == "__main__":
    app()

