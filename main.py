import os

from src.data_loader import load_raw_records
from src.preprocessing import preprocess_records
from src.modeling import train_bertopic, save_model, save_topic_assignments
from src.visualization import visualize_barchart, visualize_hierarchy, visualize_topics

def main():
    # Пути
    raw_dir = os.path.join("data", "raw")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка и препроцессинг
    records = load_raw_records(raw_dir)
    df = preprocess_records(records)
    df.to_csv(os.path.join("data", "processed", "preprocessed.csv"), index=False, encoding="utf-8")
    print("Сохранено предобработанное: data/processed/preprocessed.csv")

    # Подготовка документов
    docs_model = df["lemmas_filtered"].apply(lambda lst: " ".join(lst)).tolist()
    docs_label = df["cleaned_text"].tolist()

    # Обучение модели
    model, topics, probs = train_bertopic(docs_model)
    save_model(model)

    # Сохранение результатов
    save_topic_assignments(topics, probs, docs_label, os.path.join(output_dir, "topic_assignments.csv"))

    # Визуализация
    model.generate_topic_labels()
    visualize_barchart(model, top_n=10, output_dir=output_dir)
    visualize_hierarchy(model, output_dir=output_dir)
    visualize_topics(model, output_dir=output_dir)

if __name__ == "__main__":
    main()
