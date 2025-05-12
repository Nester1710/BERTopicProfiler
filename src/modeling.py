import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

def train_bertopic(docs: list[str]) -> tuple:
    # Эмбеддинговая модель для русского и других языков
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Векторизация с биграммами и фильтрацией редких слов
    vectorizer_model = CountVectorizer(
        ngram_range=(1),
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

def save_topic_assignments(topics, probs, docs: list[str], result_path: str, stats_path: str = None):
    from bertopic import BERTopic
    model = BERTopic.load("bertopic_model")
    labels = model.generate_topic_labels()
    label_map = {i: "_".join(label.split("_")[1:]).replace("_", " ")
    for i, label in enumerate(labels)}

    # Итоговая таблица по документам
    df_result = pd.DataFrame({
        "Topic": topics,
        "Topic_Label": [label_map.get(t) for t in topics],
        "Probability": [p.max() if p is not None else None for p in probs],
        "cleaned_text": docs,
    })
    df_result.to_csv(result_path, index=False, encoding="utf-8")

    # Статистика по темам
    if stats_path and model:
        df_stats = model.get_topic_info()
        df_stats = df_stats[["Topic", "Count", "Name"]].rename(columns={"Name": "Topic_Label"})
        df_stats["Top Terms"] = df_stats["Topic"].apply(lambda i: ", ".join([term for term, _ in model.get_topic(i)]))
        df_stats.to_csv(stats_path, index=False, encoding="utf-8")
