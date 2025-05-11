from collections import Counter
import pandas as pd

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    К DataFrame (с колонками processed_text и lemmas) добавляет:
      num_words, num_chars, avg_word_length, has_hashtag,
      top_5_unigrams, top_5_bigrams
    """
    df = df.copy()
    df["num_words"]       = df["lemmas"].apply(len)
    df["num_chars"]       = df["processed_text"].str.len()
    df["avg_word_length"] = df["num_chars"] / df["num_words"]
    df["has_hashtag"]     = df["original_text"].str.contains("#").astype(int)

    df["top_5_unigrams"] = df["lemmas"].apply(
        lambda toks: Counter(toks).most_common(5)
    )
    df["top_5_bigrams"] = df["lemmas"].apply(
        lambda toks: Counter(
            [" ".join(toks[i:i+2]) for i in range(len(toks)-1)]
        ).most_common(5)
    )
    return df
