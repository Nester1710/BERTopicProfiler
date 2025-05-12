import re
import emoji
import pandas as pd
from pymorphy2 import MorphAnalyzer
import spacy
from collections import Counter
from tqdm import tqdm
import inspect
from inspect import getfullargspec

def getargspec(func):
    spec = getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults

inspect.getargspec = getargspec

# Инициализация
morph = MorphAnalyzer()
nlp = spacy.blank("ru")
SPACY_STOPWORDS = nlp.Defaults.stop_words

def clean_text(text: str) -> str:
    """
    Очищает текст от ссылок, эмодзи, тегов, лишних символов.
    """
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\.\S+|https\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)  # Telegram club/user ссылки
    text = re.sub(r"[^\w\sа-яА-ЯёЁ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def is_informative_token(token: str) -> bool:
    """
    Отсекает мусорные токены:
    - слишком короткие;
    - полностью цифровые;
    - одиночные символы.
    """
    return len(token) > 2 and not token.isdigit()

def preprocess_records(records: list[dict]) -> pd.DataFrame:
    """
    Универсальный препроцессинг для тематического моделирования.
    Возвращает DataFrame с колонками:
      - cleaned_text: строка без ссылок и мусора
      - tokens: список слов
      - lemmas: все леммы без дубликатов
      - lemmas_filtered: леммы без стоп-слов и числового шума
      - lemmas_with_counts: список (лемма, count), где count > 1
    """
    rows = []

    for rec in tqdm(records, desc="Preprocessing", unit="doc"):
        text = rec.get("message")
        if not isinstance(text, str):
            continue

        # 1. Очистка текста
        cleaned = clean_text(text)

        # 2. Токенизация + отбор
        tokens = [t for t in cleaned.split() if is_informative_token(t)]
        if len(tokens) < 5:
            continue

        # 3. Лемматизация
        lemmas = [morph.parse(t)[0].normal_form for t in tokens]

        # 4. Фильтрация лемм
        lemmas_filtered = [lm for lm in lemmas if lm not in SPACY_STOPWORDS and is_informative_token(lm)]
        if len(lemmas_filtered) < 5:
            continue

        # 5. Частотные леммы
        freqs = Counter(lemmas_filtered)
        lemmas_with_counts = [(lm, cnt) for lm, cnt in freqs.items() if cnt > 1]

        # 6. Сбор строки
        rows.append({
            "cleaned_text":        cleaned,
            "tokens":              tokens,
            "lemmas":              list(set(lemmas)),
            "lemmas_filtered":     lemmas_filtered,
            "lemmas_with_counts":  lemmas_with_counts
        })

    return pd.DataFrame(rows)
