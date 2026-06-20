from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import pandas as pd
from nlp_rag_proj.clean import normalize_text
from nlp_rag_proj.io import load_bbc_csv

ENGLISH_STOP_WORDS: frozenset

def build_vectorizer() -> TfidfVectorizer:

    # Dodaje własne, sprawdzone w analize stop-słowa
    vectorizer = TfidfVectorizer(
        stop_words = list(ENGLISH_STOP_WORDS.union({"s", "-", "said", "new"})),
        #preprocessor=text_normalization
        )
    
    return vectorizer

def prepare_dataset(df: pd.DataFrame, split_test = True) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

    if split_test:
        X_train, X_test, y_train, y_test  = train_test_split(
            df["text"],
            df["category"],
            train_size=0.8,
            random_state=42,
            stratify=df["category"]
            )
    else:
        X_train = df["text"]
        y_train = df["category"]
        X_test = pd.Series(dtype=str)
        y_test = pd.Series(dtype=str)

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    
    df = load_bbc_csv(set_type="train")
    X_train, X_test, y_train, y_test = prepare_dataset(df, split_test=True)
    
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print(vectorizer.get_params()["preprocessor"] == normalize_text)
    
    for example_number in range(1): 
        
        raw_example = X_train.iloc[example_number]
        
        tfidf_example = X_train_tfidf[example_number]
        
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = tfidf_example.nonzero()[1]
        
        print("\n================ TEST TRANSFORMACJI DOKUMENTU =================\n")
        print("--- [PRZED] Surowy tekst z X_train (pierwsze 300 znaków) ---")
        print(raw_example[:1000] + "..." if len(raw_example) > 1000 else raw_example)
        print(f"\nKategoria docelowa: {y_train.iloc[example_number]}")
        
        print("\n--- [PO] Wyekstrahowane tokeny i ich wagi TF-IDF (Top 15 najsilniejszych) ---")
        print(f"{'Token / N-gram':<35} | {'Waga TF-IDF':<10}")
        print("-" * 50)
        
        # Sortowanie tokenów według wagi malejąco, aby zobaczyć najważniejsze słowa
        extracted_features = [(feature_names[idx], tfidf_example[0, idx]) for idx in nonzero_indices]
        extracted_features.sort(key=lambda x: x[1], reverse=True)

        for token, weight in extracted_features[:15]:
            print(f"{token:<35} | {weight:.4f}")
        print(f"\nŁącznie unikalnych tokenów w tym dokumencie: {len(nonzero_indices)}")
        print("================================================================")