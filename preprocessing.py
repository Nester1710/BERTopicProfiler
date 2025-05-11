import os
import json
import re
import emoji
import pandas as pd
import inspect
from inspect import getfullargspec
import pymorphy2
import spacy
from collections import Counter

# Патчим inspect.getargspec для Python ≥3.11, чтобы pymorphy2 заработал
def getargspec(func):
    spec = getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults
inspect.getargspec = getargspec

# Пути
nlp = spacy.blank("ru")
SPACY_STOPWORDS = nlp.Defaults.stop_words
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Datasets')

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Не найдена папка Datasets: {DATA_DIR}")

# Инициализируем MorphAnalyzer
morph = pymorphy2.MorphAnalyzer()

def load_json_array(path: str) -> list:
    """
    Надёжно извлекает объекты из большого JSON-массива {…}.
    """
    text = open(path, "r", encoding="utf-8").read()
    start = text.find('[')
    end   = text.rfind(']')
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Файл не содержит JSON-массив в квадратных скобках")
    inner = text[start+1:end]
    records = []
    depth = 0
    obj_start = None
    for i, ch in enumerate(inner):
        if ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_str = inner[obj_start:i+1]
                try:
                    records.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return records

def clean_text(text: str) -> str:
    """
    Очищает текст: убирает эмодзи, ссылки, телеграм-метки, хэштеги и спецсимволы.
    """
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^\w\sа-яА-ЯёЁ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Лемматизирует список токенов через pymorphy2 и отфильтровывает стоп-слова spaCy.
    """
    # 1) лемматизация
    lemmas = [morph.parse(w)[0].normal_form for w in tokens]
    # 2) фильтрация стоп-слов и пустых строк
    lemmas = [lm for lm in lemmas if lm and lm not in SPACY_STOPWORDS]
    return lemmas

def get_top_ngrams(tokens: list[str], n: int = 1, top_n: int = 5) -> list[tuple]:
    """
    Возвращает топ-N униграмм или биграмм.
    """
    if n == 1:
        grams = tokens
    else:
        grams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    from collections import Counter
    return Counter(grams).most_common(top_n)

def preprocess_file(filepath: str) -> pd.DataFrame:
    """
    Обрабатывает один JSON-файл:
      - извлекает message,
      - чистит текст,
      - токенизирует,
      - лемматизирует (и убирает стоп-слова),
      - пересчитывает processed_text и признаки,
      - собирает топы уни- и биграмм.
    """
    raw = load_json_array(filepath)
    rows = []

    for rec in raw:
        text = rec.get("message")
        if not isinstance(text, str):
            continue

        # очистка и токенизация
        cleaned = clean_text(text)
        tokens  = cleaned.split()
        if len(tokens) < 5:
            continue

        # лемматизация + фильтрация стоп-слов
        lemmas = lemmatize_tokens(tokens)
        if len(lemmas) < 5:
            continue  # после фильтрации могло стать слишком мало слов

        # пересобираем processed_text из лемм
        processed_text = " ".join(lemmas)
        num_words      = len(lemmas)
        num_chars      = len(processed_text)

        # топы n-грамм по леммам
        def top_ngrams(ngr: int):
            seq = [' '.join(lemmas[i:i+ngr]) for i in range(len(lemmas)-ngr+1)]
            return Counter(seq).most_common(5)

        rows.append({
            "original_text":    text,
            "processed_text":   processed_text,
            "tokens":           tokens,
            "lemmas":           lemmas,
            "num_words":        num_words,
            "num_chars":        num_chars,
            "avg_word_length":  num_chars / num_words,
            "has_hashtag":      int("#" in text),
            "top_5_unigrams":   top_ngrams(1),
            "top_5_bigrams":    top_ngrams(2),
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".json"):
            continue
        path_in = os.path.join(DATA_DIR, fname)
        print(f"Processing {fname} …")
        df_out = preprocess_file(path_in)
        out_name = fname.rsplit(".", 1)[0] + "_preprocessed.csv"
        path_out = os.path.join(DATA_DIR, out_name)
        df_out.to_csv(path_out, index=False, encoding="utf-8")
        print(f" → Saved {out_name} ({len(df_out)} rows)")
