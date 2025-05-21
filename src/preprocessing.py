# preprocessing.py

import inspect
from collections import namedtuple

# Заглушка для inspect.getargspec (нужна pymorphy2)
ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "varkw", "defaults"])
if not hasattr(inspect, "getargspec"):
    def getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec

import re
from pathlib import Path
from typing import Union, Set, List

import pandas as pd
from tqdm import tqdm
import pymorphy2
import multiprocessing
from functools import partial

from data_loader import DataLoader

# Инициализация морфо-анализатора на каждый процесс
def init_worker():
    global morph
    morph = pymorphy2.MorphAnalyzer()

# Лемматизация и фильтрация токенов, выполняется последовательно в процессе
def normalize_and_filter(toks: List[str], stopwords: Set[str]) -> str:
    lemmas = []
    for w in toks:
        if w and w not in stopwords:
            lm = morph.parse(w)[0].normal_form
            if lm and lm not in stopwords:
                lemmas.append(lm)
    return " ".join(lemmas)

# Очистка и лемматизация одной серии без вложенного пула
def clean_text_vectorized(series: pd.Series, stopwords: Set[str]) -> pd.Series:
    # 1) векторная очистка
    s = series.fillna("").str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", "", regex=True)
    s = s.str.replace(r"[@#]\w+", "", regex=True)
    s = s.str.replace(r"\+?\d[\d\-\s]{7,}\d", "", regex=True)
    s = s.str.replace(r"\d+", "", regex=True)
    s = s.str.replace(r"[^\w\s]", "", regex=True)

    # 2) токенизация
    tokens_list = s.str.split().tolist()

    # 3) последовательная лемматизация в этом процессе
    clean_list = [normalize_and_filter(toks, stopwords) for toks in tokens_list]

    return pd.Series(clean_list, index=series.index)

# Обработка одного файла (будет запускаться в пуле)
def _process_file(path: Path, output_dir: Path, stopwords: Set[str]):
    df = pd.read_csv(path)
    if not {"text", "collection"}.issubset(df.columns):
        return

    df["clean_text"] = clean_text_vectorized(df["text"], stopwords)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    out = df[["collection", "clean_text", "word_count"]].copy()
    out.insert(0, "id", range(1, len(out) + 1))

    dst = output_dir / f"{path.stem}_preprocessed.csv"
    out.to_csv(dst, index=False, encoding="utf-8")

def preprocess_datasets(raw_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        stopwords: Set[str]) -> None:
    """
    Параллельная обработка по файлам:
      - пул процессов инициализируется один раз
      - каждый процесс загружает, очищает, лемматизирует и сохраняет свой файл
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = DataLoader(raw_dir, output_dir).list_raw_datasets()
    cpu = multiprocessing.cpu_count()

    with multiprocessing.Pool(cpu, initializer=init_worker) as pool:
        func = partial(_process_file, output_dir=output_dir, stopwords=stopwords)
        list(tqdm(pool.imap(func, files), total=len(files), desc="Preprocessing"))
