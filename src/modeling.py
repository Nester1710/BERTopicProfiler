import os
import pandas as pd
from collections import Counter
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


def train_bertopic(docs: list[str]) -> tuple:
    """
    Обучает BERTopic на списке документов (docs).
    Возвращает: (модель, список тем, список вероятностей).
    """
    # Эмбеддинговая модель для русского и других языков
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Векторизация с биграммами и фильтрацией редких слов
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        stop_words=None
    )

    # Инициализация BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=10,
        nr_topics="auto",
        language="multilingual",
        verbose=True
    )

    # Обучение
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def save_topic_assignments(topics, probs, docs: list[str], output_path: str):
    """
    Сохраняет CSV с колонками: Document, Topic, Probability, Topic_Label
    """
    # Генерируем читаемые метки тем
    # Генерируем читаемые метки тем
    topic_model = BERTopic.load("bertopic_model") if os.path.exists("bertopic_model") else None
    labels = topic_model.generate_topic_labels() if topic_model else []
    label_map = {i: label for i, label in enumerate(labels)}

    df = pd.DataFrame({
        "Document": docs,
        "Topic": topics,
        "Probability": [p.max() if p is not None else None for p in probs],
    })

    if label_map:
        df["Topic_Label"] = df["Topic"].map(label_map)

    df.to_csv(output_path, index=False, encoding="utf-8")


def save_model(model: BERTopic, path: str = "bertopic_model"):
    """
    Сохраняет обученную модель на диск.
    """
    model.save(path)
