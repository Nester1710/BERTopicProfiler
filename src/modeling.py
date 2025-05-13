import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def train_bertopic(docs: list[str]) -> tuple:
    docs = [doc if isinstance(doc, str) else "" for doc in docs]
    docs = [doc for doc in docs if doc.strip()]

    # Инициализация моделей
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    stop_words = get_stop_words('russian')
    vectorizer = CountVectorizer(stop_words=stop_words)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        language='russian'
    )

    # Этап 1: извлечение эмбеддингов
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    # Этап 2: обучение модели
    with tqdm(total=1, desc='Fitting BERTopic') as bar:
        topics, probs = topic_model.fit_transform(docs, embeddings)
        bar.update(1)

    # Этап 3: генерация меток тем
    topic_model.generate_topic_labels()

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
