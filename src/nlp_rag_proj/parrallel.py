import pandas as pd
import time
from nlp_rag_proj.io import load_bbc_csv # do testów
from nlp_rag_proj.clean import normalize_text
import os

from pathlib import Path


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

def sequential(ext_path: Path) -> None:

    for filename in os.listdir(ext_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(ext_path, filename)
            with open(full_path, "r", encoding="utf-8") as plik:
                zawartosc = plik.read()
    pass

def parrallel(ext_path: Path) -> None:
    pass
    

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")
    ext_path = Path.cwd() / "data" / "extracted"
    extract_random_articles(df, 500, ext_path)

    sequential()
    parrallel()