# src/preprocessing.py

import inspect
from inspect import getfullargspec
def getargspec(func):
    spec = getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults
import inspect
inspect.getargspec = getargspec

import pandas as pd
from pymorphy2 import MorphAnalyzer
import spacy
from functools import lru_cache
from tqdm import tqdm

from .utils import clean_text

# Настройка морфоанализатора и стоп-слов spaCy
morph = MorphAnalyzer()
nlp = spacy.blank("ru")
SPACY_STOPWORDS = nlp.Defaults.stop_words

@lru_cache(maxsize=None)
def lemmatize_word(word: str) -> str:
    """
    Возвращает нормальную форму слова через pymorphy2,
    результат запоминается в кэше.
    """
    return morph.parse(word)[0].normal_form

def preprocess_records(records: list[dict]) -> pd.DataFrame:
    """
    Обрабатывает записи:
      1) чистит текст;
      2) токенизирует;
      3) лемматизирует через кэшированную функцию;
      4) фильтрует стоп-слова spaCy;
      5) формирует DataFrame с колонками:
         original_text, processed_text, tokens, lemmas
    Индикатор прогресса выводится через tqdm.
    """
    rows = []
    for rec in tqdm(records, desc="Preprocessing", unit="doc"):
        text = rec.get("message")
        if not isinstance(text, str):
            continue

        cleaned = clean_text(text)
        tokens  = cleaned.split()
        if len(tokens) < 5:
            continue

        # Лемматизация с кэшированием
        lemmas = [lemmatize_word(w) for w in tokens]
        # Фильтрация стоп-слов
        lemmas = [lm for lm in lemmas if lm and lm not in SPACY_STOPWORDS]
        if len(lemmas) < 5:
            continue

        rows.append({
            "original_text":  text,
            "processed_text": " ".join(lemmas),
            "tokens":         tokens,
            "lemmas":         lemmas
        })

    return pd.DataFrame(rows)
