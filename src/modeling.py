from pathlib import Path
from typing import List, Union

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from stop_words import get_stop_words

from analysis import analyze_model


class BERTopicTrainer:
    """
    Инициализация и батч-обучение BERTopic с автоматическим анализом.
    """

    def __init__(self) -> None:
        # Стоп-слова
        stopwords = list(dict.fromkeys(
            get_stop_words("russian") +
            get_stop_words("english")
        ))

        # Модель эмбеддингов
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # UMAP для снижения размерности
        umap_model = UMAP(
            n_neighbors=40,
            n_components=2,
            min_dist=0.0,
            low_memory=True,
            n_jobs=-1
        )

        # HDBSCAN для кластеризации
        hdbscan_model = HDBSCAN(
            min_cluster_size=50,
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.0,
            prediction_data=True,
            core_dist_n_jobs=-1
        )

        # Векторизация текста
        vectorizer_model = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 2),
            min_df=30,
            max_df=0.85
        )
        ctfidf_model = ClassTfidfTransformer()

        self.model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model
        )

    def train(self,
              docs: List[str],
              analyze: bool = True
              ) -> (List[int], np.ndarray):
        """
        1) Кодирование документов (с прогресс-баром).
        2) Обучение модели и получение тем и вероятностей.
        3) Автоматический анализ (если analyze=True).
        """
        embeddings = self.model.embedding_model.encode(
            docs,
            show_progress_bar=True
        )
        topics, probabilities = self.model.fit_transform(
            docs,
            embeddings=embeddings
        )

        if analyze:
            analyze_model(self.model, docs, topics, embeddings)

        return topics, probabilities

    def save(self, path: Union[str, Path]) -> None:
        """Сохранение модели на диск."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BERTopicTrainer":
        """Загрузка модели из директории."""
        trainer = cls()
        trainer.model = BERTopic.load(str(path))
        return trainer
