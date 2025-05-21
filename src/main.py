from pathlib import Path
import sys

from data_loader import DataLoader
from preprocessing import preprocess_datasets
from modeling import BERTopicTrainer
from stop_words import get_stop_words

# Пути к директориям
RAW_DIR = Path("../data/")
PREPROC_DIR = Path("../preprocess/")
MODELS_DIR = Path("../models/")
OUTPUTS_DIR = Path("../outputs/")


def run_preprocessing() -> None:
    """Предобработка сырых данных."""
    stopwords = set(
        get_stop_words("russian") +
        get_stop_words("english") +
        [
            "bbg", "etf", "млрд", "млн", "трлн", "михалыч",
            "bbgcrypto", "видео", "михаилович", "ура",
            "благодарить", "здравствуйте", "руб", "bloomberg"
        ]
    )
    preprocess_datasets(RAW_DIR, PREPROC_DIR, stopwords)
    print("Предобработка завершена →", PREPROC_DIR)


def run_training() -> None:
    """Обучение модели и сохранение результатов."""
    loader = DataLoader(RAW_DIR, PREPROC_DIR)
    df = loader.load_preprocessed()
    documents = df["clean_text"].tolist()

    trainer = BERTopicTrainer()
    topics, probabilities = trainer.train(documents)

    model_path = MODELS_DIR / "bertopic"
    trainer.save(model_path)
    print("Модель сохранена →", model_path)

    df["topic"] = topics
    df["probability"] = probabilities
    df["topic_name"] = df["topic"].map(trainer.model.topic_labels_)

    output_df = df[["id", "collection", "clean_text", "topic_name", "probability"]]
    OUTPUTS_DIR.mkdir(exist_ok=True)
    result_path = OUTPUTS_DIR / "topics.csv"
    output_df.to_csv(result_path, index=False, encoding="utf-8")
    print("Результаты сохранены →", result_path)


def main_menu() -> None:
    """Главное меню приложения."""
    while True:
        print("""
1) Предобработать данные
2) Обучить модель
0) Выход
""")
        choice = input("Выберите опцию: ").strip()
        if choice == "1":
            run_preprocessing()
        elif choice == "2":
            run_training()
        elif choice == "0":
            sys.exit()
        else:
            print("Неверный ввод, попробуйте снова.")


if __name__ == "__main__":
    main_menu()
