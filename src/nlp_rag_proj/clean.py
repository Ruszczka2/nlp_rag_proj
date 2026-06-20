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
            print("Running stem...")
            processed_list = [" ".join(stemmer.stemWords(text.split())) for text in texts_to_process]
            return pd.Series(processed_list, index=series.index)
        elif args.lem:
            print("Running lemmatizator...")
            docs = nlp_obj.pipe(texts_to_process, batch_size=256)
            processed_list = [" ".join([token.lemma_ for token in doc]) for doc in docs]
            return pd.Series(processed_list, index=series.index)
        
        return processed

if __name__ == "__main__":
    print(string.punctuation.replace('$',''))