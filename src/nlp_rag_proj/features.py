from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import pandas as pd
from nlp_rag_proj.clean import normalize_text

def build_vectorizer() -> TfidfVectorizer:

    # Dodaje własne, sprawdzone w analize stop-słowa
    vectorizer = TfidfVectorizer(
        stop_words = list(ENGLISH_STOP_WORDS.union({"s", "-", "said", "new"})),
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        max_features=8000
        )
    
    return vectorizer

def prepare_dataset(df: pd.DataFrame, split_test = False):

    if split_test:
        X_train = df["Text"]
        y_train = df["Category"]
        X_test = pd.Series(dtype=object)
        y_test = pd.Series(dtype=object)
    else:
        X_train, X_test, y_train, y_test  = train_test_split(
            df["Text"],
            df["Category"], 
            train_size=0.8, 
            random_state=42, 
            stratify=df["Category"]
            )

    return X_train, X_test, y_train, y_test

from nlp_rag_proj.io import load_bbc_csv

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")

    print("--- Przypadek: Cały zbiór jako trening ---")
    X_train, X_test, y_train, y_test = prepare_dataset(df, split_test=True)
    print(f"Rozmiar train: {X_train.shape[0]}, Rozmiar test: {X_test.shape[0]}")
    print(X_train.head(2))

    print("\n--- Przypadek: Podział 80/20 ---")
    X_train, X_test, y_train, y_test = prepare_dataset(df, split_test=False)
    print(f"Rozmiar train: {X_train.shape[0]}, Rozmiar test: {X_test.shape[0]}")
    print(X_train.head(2))