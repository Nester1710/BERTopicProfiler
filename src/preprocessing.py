import inspect
import re
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Set, List, Union
import multiprocessing

import pandas as pd
import pymorphy2
from tqdm import tqdm

from data_loader import DataLoader

# Заглушка для inspect.getargspec в Python>=3
ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "varkw", "defaults"])
if not hasattr(inspect, "getargspec"):
    def getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec


def init_worker() -> None:
    """Инициализация морфологического анализатора в каждом процессе."""
    global morph
    morph = pymorphy2.MorphAnalyzer()


def normalize_and_filter(tokens: List[str], stopwords: Set[str]) -> str:
    """Лемматизация токенов и фильтрация стоп-слов."""
    lemmas = []
    for token in tokens:
        if token and token not in stopwords:
            normal_form = morph.parse(token)[0].normal_form
            if normal_form and normal_form not in stopwords:
                lemmas.append(normal_form)
    return " ".join(lemmas)


def clean_text_vectorized(series: pd.Series, stopwords: Set[str]) -> pd.Series:
    """
    Векторная очистка и лемматизация без многопроцесса:
      - удаление URL, упоминаний, цифр, пунктуации
      - токенизация
      - последовательная лемматизация
    """
    s = series.fillna("").str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", "", regex=True)
    s = s.str.replace(r"[@#]\w+", "", regex=True)
    s = s.str.replace(r"\+?\d[\d\-\s]{7,}\d", "", regex=True)
    s = s.str.replace(r"\d+", "", regex=True)
    s = s.str.replace(r"[^\w\s]", "", regex=True)

    token_lists = s.str.split().tolist()
    cleaned = [
        normalize_and_filter(tokens, stopwords)
        for tokens in token_lists
    ]
    return pd.Series(cleaned, index=series.index)


def _process_file(path: Path, output_dir: Path, stopwords: Set[str]) -> None:
    """
    Обработка одного CSV-файла:
      - очистка текста
      - подсчет количества слов
      - сохранение результата
    """
    df = pd.read_csv(path, encoding="utf-8")
    if not {"text", "collection"}.issubset(df.columns):
        return

    df["clean_text"] = clean_text_vectorized(df["text"], stopwords)
    df["word_count"] = df["clean_text"].str.split().str.len()

    # Фильтрация пустых текстов
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    result = df[["collection", "clean_text", "word_count"]].copy()
    result.insert(0, "id", range(1, len(result) + 1))

    output_path = output_dir / f"{path.stem}_preprocessed.csv"
    result.to_csv(output_path, index=False, encoding="utf-8")


def preprocess_datasets(raw_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        stopwords: Set[str]) -> None:
    """
    Параллельная обработка всех файлов из raw_dir с сохранением в output_dir.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = DataLoader(raw_path, out_path).list_raw_datasets()
    cpu_count = multiprocessing.cpu_count()

    with multiprocessing.Pool(cpu_count, initializer=init_worker) as pool:
        worker = partial(_process_file, output_dir=out_path, stopwords=stopwords)
        list(tqdm(pool.imap(worker, files), total=len(files), desc="Preprocessing"))
