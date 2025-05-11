import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

def train_bertopic(docs: list[str],
                   ngram_range=(1, 2),
                   min_topic_size=20,
                   nr_topics="auto",
                   verbose=True):
    """
    Инициализирует и обучает BERTopic на списке строк docs.
    Для ngram_range создаёт CountVectorizer, затем передаёт его в модель.
    """
    # создаём кастомный векторизатор
    vectorizer_model = CountVectorizer(ngram_range=ngram_range)

    model = BERTopic(
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        verbose=verbose
    )
    topics, probs = model.fit_transform(docs)
    return model, topics, probs

def save_topic_assignments(topics, probs, docs, output_path: str):
    """
    Сохраняет CSV с колонками Document, Topic, Probability.
    """
    df = pd.DataFrame({
        "Document":    docs,
        "Topic":       topics,
        "Probability": [p.max() if p is not None else None for p in probs]
    })
    df.to_csv(output_path, index=False, encoding="utf-8")
