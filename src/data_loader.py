# data_loader.py
from pathlib import Path
from typing import List, Union

import pandas as pd


class DataLoader:
    """
    Загрузка сырых и предобработанных датасетов.
    """

    def __init__(self, raw_dir: Union[str, Path], preproc_dir: Union[str, Path]):
        self.raw_dir = Path(raw_dir)
        self.preproc_dir = Path(preproc_dir)

    def list_raw_datasets(self) -> List[Path]:
        return list(self.raw_dir.glob("*.csv"))

    def load_raw(self, path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(path)

    def list_preprocessed_datasets(self) -> List[Path]:
        return list(self.preproc_dir.glob("*.csv"))

    def load_preprocessed(self) -> pd.DataFrame:
        """
        Считывает все предобработанные CSV, удаляет пустые clean_text,
        возвращает единый DataFrame с колонками id, collection, clean_text, word_count.
        """
        dfs = []
        for p in self.list_preprocessed_datasets():
            df = pd.read_csv(p)
            if "clean_text" not in df.columns:
                continue
            df["clean_text"] = df["clean_text"].fillna("").astype(str)
            df = df[df["clean_text"].str.strip().astype(bool)]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
