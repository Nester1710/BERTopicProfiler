import os
import pandas as pd

from src.data_loader import load_raw_records
from src.preprocessing import preprocess_records
from src.modeling import train_bertopic, save_topic_assignments
from src.visualization import visualize_barchart, visualize_hierarchy, visualize_topics
from bertopic import BERTopic

CSV_DIR = "CSV"
RAW_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_PATH = "bertopic_model"

os.makedirs(CSV_DIR, exist_ok=True)

def run_preprocessing():
    records = load_raw_records(RAW_DIR)
    df = preprocess_records(records)
    df.to_csv(os.path.join(CSV_DIR, "Preprocessing.csv"), index=False, encoding="utf-8")
    print(f"✅ Предобработка завершена. Сохранено: {CSV_DIR}")

def run_training():
    df = pd.read_csv(os.path.join(CSV_DIR, "Preprocessing.csv"))
    docs_model = df["lemmas_filtered"].apply(eval).apply(lambda lst: " ".join(lst)).tolist()
    model, topics, probs = train_bertopic(docs_model)
    model.generate_topic_labels()
    model.save("bertopic_model")
    print("✅ Обучение завершено, модель сохранена.")

def run_model():
    if not os.path.exists(os.path.join(CSV_DIR, "Preprocessing.csv")) or not os.path.exists(MODEL_PATH):
        print("❌ Отсутствует модель или предобработанный текст.")
        return
    df = pd.read_csv(os.path.join(CSV_DIR, "Preprocessing.csv"))
    docs = df["lemmas_filtered"].apply(eval).apply(lambda lst: " ".join(lst)).tolist()
    docs_label = df["cleaned_text"].tolist()
    model = BERTopic.load(MODEL_PATH)
    model.generate_topic_labels()
    topics, probs = model.transform(docs)
    save_topic_assignments(
        topics,
        probs,
        docs_label,
        result_path=os.path.join(CSV_DIR, "Result.csv"),
        stats_path=os.path.join(CSV_DIR, "Statistics.csv")
    )
    visualize_barchart(model, top_n=10, output_dir=OUTPUT_DIR)
    visualize_hierarchy(model, output_dir=OUTPUT_DIR)
    visualize_topics(model, output_dir=OUTPUT_DIR)
    print("✅ Модель загружена и визуализации сохранены в папке outputs.")

def main():
    while True:
        print("\nВыберите действие:")
        print("1. Предобработка")
        print("2. Обучение модели")
        print("3. Запуск модели")
        print("4. Завершить работу")

        choice = input("Введите номер действия: ").strip()

        match choice:
            case "1":
                run_preprocessing()
            case "2":
                run_training()
            case "3":
                run_model()
            case "4":
                print("👋 Работа завершена.")
                break
            case _:
                print("❓ Неизвестная команда. Попробуйте ещё раз.")

if __name__ == "__main__":
    main()
