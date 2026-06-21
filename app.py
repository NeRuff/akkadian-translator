import streamlit as st
import torch
from transformers import ByT5Tokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Akkadian Translator", layout="centered")
st.title("Аккадский → Английский")

# Загрузка модели с правильным токенизатором
@st.cache_resource
def load_model():
    model_path = "./models/optimized/final"
    tokenizer = ByT5Tokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

def translate(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
    outputs = model.generate(**inputs, num_beams=4, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

akkadian_text = st.text_area("Введите текст на аккадском (транслитерация):", height=150)

if st.button("Перевести", type="primary"):
    if akkadian_text:
        with st.spinner("Перевод..."):
            result = translate(akkadian_text)
        st.success("Перевод готов!")
        st.text_area("Результат:", result, height=150)
    else:
        st.warning("Введите текст.")
