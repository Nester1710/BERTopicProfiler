from pathlib import Path
from typing import List, Union

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from stop_words import get_stop_words

from analysis import analyze_model  # <-- наш новый модуль


class BERTopicTrainer:
    """
    Инициализация и batch-обучение BERTopic на всем корпусе сразу,
    с автоматическим анализом после train().
    """

    def __init__(self):
        sw = list(dict.fromkeys(
            get_stop_words("russian") + get_stop_words("english")
        ))

        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        umap_model = UMAP(
            n_neighbors=40, n_components=2, min_dist=0.0,
            low_memory=True, n_jobs=-1
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=50,
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.0,
            prediction_data=True,
            core_dist_n_jobs=-1
        )
        vectorizer = CountVectorizer(
            stop_words=sw, ngram_range=(1, 2),
            min_df=30, max_df=0.85
        )
        ctfidf = ClassTfidfTransformer()

        self.model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf
        )

    def train(self,
              docs: List[str],
              analyze: bool = True
              ) -> (List[int], np.ndarray):
        """
        1) Кодируем все docs (показываем прогресс-бар)
        2) Обучаем модель
        3) Автоматически вызываем анализ (если analyze=True)
        """
        embeddings = self.model.embedding_model.encode(
            docs, show_progress_bar=True
        )
        topics, probs = self.model.fit_transform(
            docs, embeddings=embeddings
        )

        if analyze:
            analyze_model(self.model, docs, topics, embeddings)

        return topics, probs

    def save(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(exist_ok=True, parents=True)
        self.model.save(str(p))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BERTopicTrainer":
        t = cls()
        t.model = BERTopic.load(str(path))
        return t
