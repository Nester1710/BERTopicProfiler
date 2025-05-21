# preprocessing.py
import re
from pathlib import Path
from typing import Union, Set

import pandas as pd
from tqdm import tqdm

import inspect
from collections import namedtuple

ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "varkw", "defaults"])

if not hasattr(inspect, "getargspec"):
    def getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec

import pymorphy2

from data_loader import DataLoader

morph = pymorphy2.MorphAnalyzer()

def clean_text_vectorized(series: pd.Series, stopwords: Set[str]) -> pd.Series:
    """
    Vectorized cleaning + lemmatization:
      - lowercase, remove URLs/htags/phones/digits/punct
      - split → remove stopwords → lemmatize → join
    """
    s = series.fillna("").str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", "", regex=True)
    s = s.str.replace(r"[@#]\w+", "", regex=True)
    s = s.str.replace(r"\+?\d[\d\-\s]{7,}\d", "", regex=True)
    s = s.str.replace(r"\d+", "", regex=True)
    s = s.str.replace(r"[^\w\s]", "", regex=True)

    # токенизация
    tokens = s.str.split()

    # стоп-слова + лемматизация
    def normalize_and_filter(toks):
        lemmas = []
        for w in toks:
            if w and w not in stopwords:
                lem = morph.parse(w)[0].normal_form
                if lem not in stopwords:
                    lemmas.append(lem)
        return " ".join(lemmas)

    return tokens.apply(normalize_and_filter)

def preprocess_datasets(raw_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        stopwords: Set[str]) -> None:
    """
    Для каждого CSV в raw_dir:
      - загрузка целиком через DataLoader
      - очистка текста и подсчет word_count
      - сохранение id, collection, clean_text, word_count в output_dir
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    loader = DataLoader(raw_dir, output_dir)
    for src in tqdm(loader.list_raw_datasets(), desc="Preprocessing"):
        df = loader.load_raw(src)
        if "text" not in df.columns or "collection" not in df.columns:
            continue

        df["clean_text"] = clean_text_vectorized(df["text"], stopwords)
        df["word_count"] = df["clean_text"].str.split().str.len()
        df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

        out = df[["collection", "clean_text", "word_count"]].copy()
        out.insert(0, "id", range(1, len(out) + 1))

        dst = output_dir / f"{src.stem}_preprocessed.csv"
        out.to_csv(dst, index=False, encoding="utf-8")
