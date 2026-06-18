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

def prepare_dataset(df: pd.DataFrame):

    X_train, X_test, y_train, y_test  = train_test_split(df["Text"], df["Category"], train_size=0.8, random_state=42, stratify=df["Category"])

    return X_train, X_test, y_train, y_test