from pymorphy2 import MorphAnalyzer
from collections import Counter
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import inspect
import spacy
import re

from inspect import getfullargspec
def getargspec(func):
    spec = getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults
inspect.getargspec = getargspec

# Инициализация
morph = MorphAnalyzer()
nlp = spacy.blank("ru")
SPACY_STOPWORDS = nlp.Defaults.stop_words

# Очистка текста
def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)                     # Telegram club/user ссылки
    text = re.sub(r"http\S+|www\.\S+|https\S+", "", text)   # Ссылки
    text = re.sub(r"[^\w\sа-яА-ЯёЁ]", " ", text)            # Символы
    text = re.sub(r"\s+", " ", text)                        # Лишние пробелы
    return text.strip().lower()

def is_informative_token(token: str) -> bool:
    return len(token) > 3 and not token.isdigit()

def process_single_record(rec: dict) -> dict | None:
    text = rec.get("message")
    if not isinstance(text, str):
        return None

    # 1. Очистка текста
    cleaned = clean_text(text)

    # 2. Токенизация + отбор
    tokens = [t for t in cleaned.split() if is_informative_token(t)]
    if len(tokens) < 5:
        return None

    # 3. Лемматизация
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]

    # 4. Фильтрация лемм
    lemmas_filtered = [lm for lm in lemmas if lm not in SPACY_STOPWORDS and is_informative_token(lm)]
    if len(lemmas_filtered) < 5:
        return None

    # 5. Частотные леммы
    freqs = Counter(lemmas_filtered)
    lemmas_with_counts = [(lm, cnt) for lm, cnt in freqs.items() if cnt > 1]

    # 6. Сбор строки
    return {
        "cleaned_text":        cleaned,
        "tokens":              tokens,
        "lemmas":              list(set(lemmas)),
        "lemmas_filtered":     lemmas_filtered,
        "lemmas_with_counts":  lemmas_with_counts
    }

def preprocess_records(records: list[dict]) -> pd.DataFrame:
    with Pool(cpu_count() - 1) as pool:
        rows = list(tqdm(pool.imap(process_single_record, records), total=len(records), desc="Preprocessing", unit="doc"))

    # Убираем None (отфильтрованные записи)
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)
