import re
import string
from typing import Any
import pandas as pd

from nlp_rag_proj.tokenization import  stem, lemmatize

def normalize_text(text: str) -> str:

    # Sprawdzenie czy to string
    if not isinstance(text, str):
        return ""

    # Sprowadzenie do małych liter
    text = text.lower()

    # Usuwanie linków http lub www.
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Usuwanie znaków interpunkcyjnych
    text = re.sub(rf'[{re.escape(string.punctuation.replace('$',''))}]', '', text)

    # Usuwanie cyfr
    text = re.sub(r'[1234567890]', '', text)

    # Używanie jednej spacji między słowami
    text = " ".join(text.split())

    return text

def apply_nlp(series: pd.Series, args: Any | None, *, stemmer, nlp_obj) -> pd.Series:
        
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pd.Series, got: {type(series)}")
        
        processed = series.apply(normalize_text)
        texts_to_process = processed.tolist()

        if args.stem:
             stem(series, texts_to_process, stemmer)
        elif args.lem:
            lemmatize(series, texts_to_process, nlp_obj)
        
        return processed

if __name__ == "__main__":
    print(string.punctuation.replace('$',''))