from nlp_rag_proj.clean import normalize_text

        
def process_single_file(full_path: str) -> None:
    for _ in range(1):
        with open(full_path, "r", encoding="utf-8") as plik:
            text = plik.read()
            text_normalized = normalize_text(text)
            text_words = text_normalized.split()
            len(text_words)
            sorted(text_words)
            sorted(text_words, key=len, reverse=True)
    

if __name__ == "__main__":
    pass