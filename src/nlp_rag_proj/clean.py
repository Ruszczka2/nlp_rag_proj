import re
import string

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
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)

    # Usuwanie cyfr
    text = re.sub(r'[1234567890]', '', text)

    # Używanie spacji między słowami
    text = " ".join(text.split())

    return text

if __name__ == "__main__":
    pass