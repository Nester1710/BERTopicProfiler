import os
import pandas as pd

from src.data_loader       import load_raw_records
from src.preprocessing     import preprocess_records
from src.feature_extraction import add_statistical_features
from src.modeling          import train_bertopic, save_topic_assignments
from src.visualization     import visualize_barchart, visualize_hierarchy

def main():
    raw_dir       = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    output_dir    = "outputs"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка и предобработка
    records = load_raw_records(raw_dir)
    df_pre  = preprocess_records(records)
    pre_path = os.path.join(processed_dir, "preprocessed.csv")
    df_pre.to_csv(pre_path, index=False, encoding="utf-8")
    print(f"Сохранено предобработанное: {pre_path}")

    # 2. Вычисление признаков
    df_feat = add_statistical_features(df_pre)

    # 3. Тематическое моделирование
    docs = df_feat["processed_text"].tolist()
    model, topics, probs = train_bertopic(docs)

    # 4. Сохранение распределения по темам
    assign_path = os.path.join(output_dir, "topic_assignments.csv")
    save_topic_assignments(topics, probs, docs, assign_path)
    print(f"Сохранены назначения тем: {assign_path}")

    # 5. Визуализация
    bar_path = visualize_barchart(model, top_n=10, output_dir=output_dir)
    tree_path = visualize_hierarchy(model, output_dir=output_dir)
    print(f"Сохранено: {bar_path}")
    print(f"Сохранено: {tree_path}")

if __name__ == "__main__":
    main()
