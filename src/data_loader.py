from pathlib import Path
from typing import List, Union

import pandas as pd


class DataLoader:
    """
    Загрузка и объединение CSV-файлов:
      - сырые данные
      - предобработанные данные (clean_text, word_count)
    """

    def __init__(self, raw_dir: Union[str, Path], preproc_dir: Union[str, Path]):
        self.raw_dir = Path(raw_dir)
        self.preproc_dir = Path(preproc_dir)

    def list_raw_datasets(self) -> List[Path]:
        """Возвращает список CSV-файлов в raw_dir."""
        return list(self.raw_dir.glob("*.csv"))

    def load_raw(self, path: Union[str, Path]) -> pd.DataFrame:
        """Считывает один CSV-файл сырых данных."""
        return pd.read_csv(path)

    def list_preprocessed_datasets(self) -> List[Path]:
        """Возвращает список CSV-файлов в preproc_dir."""
        return list(self.preproc_dir.glob("*.csv"))

    def load_preprocessed(self) -> pd.DataFrame:
        """
        Объединяет все предобработанные CSV:
          - удаляет пустые clean_text
          - возвращает DataFrame с колонками id, collection, clean_text, word_count
        """
        dataframes = []
        for path in self.list_preprocessed_datasets():
            df = pd.read_csv(path, encoding="utf-8")
            if "clean_text" not in df.columns:
                continue

            # Заполняем NaN и фильтруем пустые строки
            df["clean_text"] = df["clean_text"].fillna("").astype(str)
            df = df[df["clean_text"].str.strip().astype(bool)]
            dataframes.append(df)

        if not dataframes:
            return pd.DataFrame(columns=["id", "collection", "clean_text", "word_count"])

        return pd.concat(dataframes, ignore_index=True)
