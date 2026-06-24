import pandas as pd
from pathlib import Path
import os

from nlp_rag_proj.io import load_bbc_csv # do testów
from nlp_rag_proj.clean import normalize_text


def extract_random_articles(df: pd.DataFrame, num_articles: int, ext_path: Path) -> None:
    
    if not isinstance(num_articles, int):
        raise TypeError(f"num_articles must be {type(int)}, but it is: {type(num_articles)}")
    
    ext_path.mkdir(parents=True, exist_ok=True)
    
    df_sampled = df.sample(n=num_articles, random_state=42)

    for index, row in df_sampled.iterrows():
        file_name = f"file{index}_{row["category"]}.txt"
        
        text_to_save = str(row['text'])
        
        with open(ext_path / file_name, "w", encoding="utf-8") as plik:
            plik.write(text_to_save)

        
def process_single_file(full_path: str) -> None:
    for _ in range(10):  # Sztuczne obciążenie z Twojego benchmarku
        with open(full_path, "r", encoding="utf-8") as plik:
            text = plik.read()
            text_normalized = normalize_text(text)
            text_words = text_normalized.split()
            count_words = len(text_words)
            text_sorted_aplhabet = sorted(text_words)
            text_sorted_length = sorted(text_words, key=len, reverse=True)
    

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")
    ext_path = Path.cwd() / "data" / "extracted"
    extract_random_articles(df, 500, ext_path)

    file_paths = [os.path.join(ext_path, f) for f in os.listdir(ext_path) if f.endswith(".txt")]

    for full_path in file_paths:
        process_single_file(full_path)