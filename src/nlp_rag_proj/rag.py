from sentence_transformers import SentenceTransformer
from nlp_rag_proj.clean import normalize_text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text_normalized = normalize_text(text)

    

    pass

def build_index(docs_dir): # → embeddingi (np. sentence-transformers/all-MiniLM-L6-v2 lokalnie)
    pass 

def retrieve(query, top_k): # → cosine similarity
    pass

def answer(query): # → sklej kontekst + prompt do LLM (API lub Ollama lokalnie)
    pass

if __name__ == "__main__":
    pass