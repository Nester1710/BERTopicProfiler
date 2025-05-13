import os
import pandas as pd

from src.data_loader import load_raw_records
from src.preprocessing import preprocess_records
from src.modeling import train_bertopic, save_topic_assignments
from src.visualization import visualize_barchart, visualize_hierarchy, visualize_topics

CSV_DIR = "CSV"
RAW_DIR = "data"
OUTPUT_DIR = "outputs"

def run_preprocessing():
    records = load_raw_records(RAW_DIR)
    df = preprocess_records(records)
    df.to_csv(os.path.join(CSV_DIR, "Preprocess.csv"), index=False, encoding="utf-8")
    print(f"Предобработка завершена. Сохранено: {CSV_DIR}")

def run_training():
    df = pd.read_csv(os.path.join(CSV_DIR, "Preprocess.csv"))
    docs = df["cleaned_text"].tolist()
    model, topics, probs = train_bertopic(docs)
    save_topic_assignments(
        model,
        topics,
        probs,
        docs,
        result_path=os.path.join(CSV_DIR, "Result.csv"),
        stats_path=os.path.join(CSV_DIR, "Statistics.csv")
    )
    visualize_barchart(model, top_n=8, output_dir=OUTPUT_DIR)
    visualize_hierarchy(model, output_dir=OUTPUT_DIR)
    visualize_topics(model, output_dir=OUTPUT_DIR)
    print("Модель загружена и визуализации сохранены в папке outputs.")

def main():
    while True:
        print("\nВыберите действие:")
        print("1. Предобработка")
        print("2. Обучение модели")
        print("3. Завершение работы")

        choice = input("Введите номер действия: ").strip()

        match choice:
            case "1":
                run_preprocessing()
            case "2":
                run_training()
            case "3":
                print("Работа завершена.")
                break
            case _:
                print("Неизвестная команда. Попробуйте ещё раз.")

if __name__ == "__main__":
    main()
