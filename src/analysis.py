# analysis.py
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import silhouette_score, silhouette_samples


def analyze_model(model,
                  docs: list[str],
                  topics: list[int],
                  embeddings: np.ndarray,
                  top_n_words: int = 10):
    """
    1) Считает global и per-topic Silhouette Score (исключая topic=-1)
    2) Строит Gensim-словарь и корпус, считает CoherenceModel(c_v) для каждой темы
    3) Сохраняет метрики в CSV
    4) Вызывает визуализации BERTopic и сохраняет HTML
    """
    out = (Path("../analysis/"))
    out.mkdir(parents=True, exist_ok=True)

    # --- 0) Процент шума ---
    total = len(topics)
    noise = sum(1 for t in topics if t == -1)
    noise_pct = noise / total * 100

    # --- 1) Silhouette ---
    mask = np.array(topics) >= 0
    labels = np.array(topics)[mask]
    embs = embeddings[mask]
    sil_global = silhouette_score(embs, labels)
    sil_samples = silhouette_samples(embs, labels)
    sil_per_topic = {
        t: float(sil_samples[labels == t].mean())
        for t in np.unique(labels)
    }

    # --- 2) Topic Coherence via Gensim ---
    # токенизируем уже очищенные тексты
    tokenized = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    # формируем словарь топ-n слов каждой темы
    topic_words = []
    for t in sorted(model.get_topic_freq()["Topic"]):
        if t == -1:
            continue
        words = [w for w, _ in model.get_topic(t)[:top_n_words]]
        topic_words.append(words)

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        corpus=corpus,
        dictionary=dictionary,
        coherence='c_v'
    )
    coh_scores = cm.get_coherence_per_topic()  # список по темам
    mean_coherence = float(np.mean(coh_scores))

    # --- 3) Save metrics ---
    pd.DataFrame([{
        "global_silhouette": sil_global,
        "mean_coherence": mean_coherence,
        "noise_percent": noise_pct
    }]).to_csv(out / "metrics_global.csv", index=False)

    pd.DataFrame({
        "topic": [t for t in np.unique(labels)],
        "silhouette": [sil_per_topic[t] for t in np.unique(labels)],
        "coherence": coh_scores
    }).to_csv(out / "metrics_per_topic.csv", index=False)

    # --- 4) Visualizations ---
    model.visualize_hierarchy().write_html(out / "hierarchy.html")
    model.visualize_barchart(top_n_topics=8).write_html(out / "barchart.html")
    model.visualize_topics().write_html(out / "intertopic.html")
