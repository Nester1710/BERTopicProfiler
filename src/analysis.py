from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import silhouette_score, silhouette_samples


def analyze_model(model,
                  docs: list[str],
                  topics: list[int],
                  embeddings: np.ndarray,
                  top_n_words: int = 10) -> None:
    """
    1) Вычисляет глобальный и per-topic Silhouette Score (без учета шума, topic==-1).
    2) Строит Gensim-словарь и корпус, рассчитывает CoherenceModel (c_v) для каждой темы.
    3) Сохраняет метрики в CSV-файлы.
    4) Генерирует HTML-визуализации модели.
    """
    output_dir = Path("../analysis/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0) Процент шумовых документов
    total_docs = len(topics)
    noise_count = sum(1 for t in topics if t == -1)
    noise_percent = noise_count / total_docs * 100

    # 1) Silhouette Score (без учета шумовых)
    mask = np.array(topics) >= 0
    labels = np.array(topics)[mask]
    filtered_embs = embeddings[mask]

    sil_global = silhouette_score(filtered_embs, labels)
    sil_samples = silhouette_samples(filtered_embs, labels)
    sil_per_topic = {
        topic: float(sil_samples[labels == topic].mean())
        for topic in np.unique(labels)
    }

    # 2) Topic Coherence через Gensim
    tokenized_docs = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]

    topic_words = []
    for topic in sorted(model.get_topic_freq()["Topic"]):
        if topic == -1:
            continue
        top_words = [word for word, _ in model.get_topic(topic)[:top_n_words]]
        topic_words.append(top_words)

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        corpus=corpus,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence_scores = coherence_model.get_coherence_per_topic()
    mean_coherence = float(np.mean(coherence_scores))

    # 3) Сохранение метрик
    global_metrics = pd.DataFrame([{
        "global_silhouette": sil_global,
        "mean_coherence": mean_coherence,
        "noise_percent": noise_percent
    }])
    global_metrics.to_csv(output_dir / "metrics_global.csv", index=False)

    per_topic_metrics = pd.DataFrame({
        "topic": np.unique(labels),
        "silhouette": [sil_per_topic[t] for t in np.unique(labels)],
        "coherence": coherence_scores
    })
    per_topic_metrics.to_csv(output_dir / "metrics_per_topic.csv", index=False)

    # 4) Генерация визуализаций
    model.visualize_hierarchy().write_html(output_dir / "hierarchy.html")
    model.visualize_barchart(top_n_topics=8).write_html(output_dir / "barchart.html")
    model.visualize_topics().write_html(output_dir / "intertopic.html")
