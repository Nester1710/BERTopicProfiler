import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import umap
import hdbscan

from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    # Силуэт не определён, если <2 кластеров
    if unique_labels.shape[0] < 2:
        return {"overall": None, "per_cluster": {}}
    # общий коэффициент
    overall_score = round(silhouette_score(embeddings, labels), 4)
    # коэффициенты для каждого объекта
    sample_scores = silhouette_samples(embeddings, labels)
    # усредняем по кластерам
    per_cluster = {}
    for lab in unique_labels:
        mask = labels == lab
        per_cluster[int(lab)] = round(float(sample_scores[mask].mean()), 4)
    return {
        "overall": float(overall_score),
        "per_cluster": per_cluster
    }

def train_bertopic(docs: list[str]) -> tuple:
    docs = [doc if isinstance(doc, str) else "" for doc in docs]
    docs = [doc for doc in docs if doc.strip()]

    # 1) Эмбеддинги
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # — можно попробовать более мощные модели:
    # embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    # embedding_model = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')

    # 2) Стоп-слова и n-граммы
    russian_sw = get_stop_words('russian')
    english_sw = get_stop_words('english')
    stop_words = list(dict.fromkeys(russian_sw + english_sw))

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),    # биграммы + униграммы
        min_df=5,              # игнорировать слишком редкие токены
        max_df=0.85            # убрать слишком частые «пустые» слова
    )

    # 3) UMAP для снижения размерности перед кластеризацией
    umap_model = umap.UMAP(
        n_neighbors=15,  # меняйте: 5–50
        n_components=5,  # обычно 2–15
        min_dist=0.0,  # [0.0–0.5], чем меньше — тем плотнее кластеры
        metric='cosine',
        random_state=42
    )

    # 4) HDBSCAN для собственно кластеризации
    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=12,  # минимум точек в кластере
        min_samples=13,  # более жёсткий критерий «шумности»
        cluster_selection_epsilon=0.2,  # позволяет «склеивать» близкие плотности
        metric='euclidean',
        prediction_data=True
    )

    # 5) Собираем BERTopic с новыми параметрами
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=True,
        verbose=True
    )

    # Этап 1: извлечение эмбеддингов
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Этап 2: обучение модели
    with tqdm(total=1, desc='Fitting BERTopic') as bar:
        topics, probs = topic_model.fit_transform(docs, embeddings)
        bar.update(1)

    # Этап 3: генерация меток тем
    topic_model.generate_topic_labels()

    # Этап 4: вычисление коэффициента силуэта
    scores = compute_silhouette(embeddings, np.array(topics))
    print("Overall silhouette:", scores["overall"])
    print("Per-cluster silhouette:", scores["per_cluster"])

    # Этап 5: вычисление процента некластеризованных тем
    total_docs = len(topics)
    noise_docs = np.sum(np.array(topics) == -1)
    percent_noise = noise_docs / total_docs * 100
    print(f"Unclustered documents (topic -1): {noise_docs}/{total_docs} "
          f"({percent_noise:.2f}%)")

    return topic_model, topics, probs

def save_topic_assignments(
    topic_model,
    topics: list[int],
    probs,
    docs: list[str],
    result_path: str,
    stats_path: str = None
) -> None:

    # Составляем таблицу с результатами
    rows = []
    for doc, topic, prob in zip(docs, topics, probs):
        label = topic_model.topic_labels_.get(topic, '')
        rows.append({
            'Topic': topic,
            'Topic_Label': label,
            'Probability': prob,
            'cleaned_text': doc
        })
    pd.DataFrame(rows).to_csv(result_path, index=False)

    # Генерируем статистику по темам
    info = topic_model.get_topic_info()
    stats = []
    for _, row in info.iterrows():
        topic_id = row.Topic
        if topic_id == -1:
            continue  # пропускаем шум
        top_terms = [term for term, _ in topic_model.get_topic(topic_id)]
        stats.append({
            'Topic': topic_id,
            'Topic_Label': row.Name,
            'Count': row.Count,
            'Top_Terms': ', '.join(top_terms)
        })
    pd.DataFrame(stats).to_csv(stats_path, index=False)
