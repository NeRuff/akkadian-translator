# Akkadian to English Neural Machine Translation

Student: Германов Роман
Group: 972407

Results

Metric: chrF++ = 20.30
Metric: BLEU = 1.15

Techniques Implemented

- Orthography normalization
- Beam search decoding (num_beams=4)
- Full metric suite (BLEU, chrF++)
- Docker containerization
- Logging with Loguru
- CLI with Typer
- Poetry dependency management
- Git workflow (dev/main branches)

How to Run

Train:
python model.py train --dataset data/train.csv

Translate:
python model.py predict "šarrum ana ālim illik"

Kaggle Submission:
python model.py predict-file data/test.csv

Docker:
docker build -t akkadian-translator .
docker run --rm akkadian-translator python model.py predict "šarrum ana ālim illik"

Project Structure:
akkadian-translator/
├── model.py
├── train.py
├── predict.py
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yaml
├── scripts/
├── notebooks/
├── data/
└── logs/

Resources:
- Base model: google/byt5-small
- Dataset: Kaggle Deep Past Initiative
- Framework: Hugging Face Transformers
