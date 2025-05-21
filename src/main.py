# main.py
from pathlib import Path
import sys

from data_loader import DataLoader
from preprocessing import preprocess_datasets
from modeling import BERTopicTrainer
from stop_words import get_stop_words

RAW_DIR     = Path("../data/")
PREPROC_DIR = Path("../preprocess/")
MODELS_DIR  = Path("../models/")
OUTPUTS_DIR = Path("../outputs/")


def run_preprocessing():
    sw = set(get_stop_words("russian") + get_stop_words("english") +
             ["bbg", "etf", "млрд", "млн", "трлн", "михалыч", "bbgcrypto", "видео", "михаилович", "ура", "благодарить",
              "здравствуйте", "руб", "bloomberg"])
    preprocess_datasets(RAW_DIR, PREPROC_DIR, sw)
    print("Preprocessing done →", PREPROC_DIR)


def run_train():
    loader = DataLoader(RAW_DIR, PREPROC_DIR)
    df = loader.load_preprocessed()
    docs = df["clean_text"].tolist()

    trainer = BERTopicTrainer()
    topics, probs = trainer.train(docs)

    MODEL_PATH = MODELS_DIR / "bertopic"
    trainer.save(MODEL_PATH)
    print("Model saved →", MODEL_PATH)

    df["topic"] = topics
    df["probability"] = probs
    df["topic_name"] = df["topic"].map(trainer.model.topic_labels_)

    out = df[["id", "collection", "clean_text", "topic_name", "probability"]]
    OUTPUTS_DIR.mkdir(exist_ok=True)
    dst = OUTPUTS_DIR / "topics.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    print("Results saved →", dst)


def menu():
    while True:
        print("""
1) Preprocess data
2) Train model
0) Exit
""")
        c = input("Choice: ").strip()
        if c == "1":
            run_preprocessing()
        elif c == "2":
            run_train()
        elif c == "0":
            sys.exit()
        else:
            print("Invalid, try again.")


if __name__ == "__main__":
    menu()
