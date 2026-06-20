import spacy
import torch
import Stemmer

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

def stem(text: str, stem_object) -> str:
    pass


def lemmatize(text: str, lem_object: spacy.Language) -> str:

    if not isinstance(text, str):
        raise TypeError(f"text must be {type("")}, but it is: {type(text)}")

    doc = lem_object(text)

    return " ".join([token.lemma_ for token in doc])

if __name__ == "__main__":
    init_gpu()

    print("\nLoading lemmatization model...")
    nlp = spacy.load("en_core_web_trf")
    text = "The cats are sitting on the windows."
    lemm_text = lemmatize(text, nlp)
    print(lemm_text)