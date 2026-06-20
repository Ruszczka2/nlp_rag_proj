import spacy
import torch
import Stemmer
import pandas as pd

def init_gpu() -> bool:
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        try:
            spacy_using_gpu = spacy.require_gpu()
            print(f"Is spaCy using GPU: {spacy_using_gpu}")
            return spacy_using_gpu
        except ValueError as e:
            print(f"Init error GPU in spaCy: {e}")
            return False
    print("PyTorch did not detect GPU.")
    return False

def tokenize(text: str) -> list[str]:
    pass

def stem(series: pd.Series, texts_to_process: list[str], stemmer) -> str:
    print("Running stem...")
    processed_list = [" ".join(stemmer.stemWords(text.split())) for text in texts_to_process]
    return pd.Series(processed_list, index=series.index)


def lemmatize(series: pd.Series, texts_to_process: list[str], nlp_obj: spacy.Language) -> str:
    print("Running lemmatizator...")
    docs = nlp_obj.pipe(texts_to_process, batch_size=256)
    processed_list = [" ".join([token.lemma_ for token in doc]) for doc in docs]
    return pd.Series(processed_list, index=series.index)

if __name__ == "__main__":
    init_gpu()

    print("\nLoading lemmatization model...")
    nlp = spacy.load("en_core_web_trf")
    text = "The cats are sitting on the windows."
    lemm_text = lemmatize(text, nlp)
    print(lemm_text)