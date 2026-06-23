import pandas as pd
import time
from nlp_rag_proj.io import load_bbc_csv # do testów


def extract_random_articles(df: pd.DataFrame, num_articles: int) -> pd.Series:
    
    if not isinstance(num_articles, int):
        raise TypeError(f"num_articles must be {type(int)}, but it is: {type(num_articles)}")
    
    df = df["text"]
    df = df.sample(n=num_articles, random_state=42)

    return df

def sequential(series: pd.Series) -> None:
    time

def parrallel(series: pd.Series) -> None:
    pass
    

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")
    series = extract_random_articles(df, 500)

    sequential(series)
    parrallel(series)