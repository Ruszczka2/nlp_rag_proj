from nlp_rag_proj.clean import normalize_text
from nlp_rag_proj.io import load_bbc_csv
from pathlib import Path
import pandas as pd
import faiss

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:

    if chunk_size < overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")
    
    if not isinstance(text, (str, pd.Series, pd.DataFrame)):
        raise TypeError(f"input text is wrong type: ({type(text)})")

    if not isinstance(chunk_size, int) or not isinstance(overlap, int):
        raise TypeError(f"inputs must be int: chunk_size ({type(chunk_size)}), overlap ({type(overlap)})")
    
    text_normalized = normalize_text(text, rag=True)
    text_list = text_normalized.split()

    step = chunk_size - overlap
    chunks = [' '.join(text_list[i:i+chunk_size]) for i in range(0, len(text_list), step)]
    return chunks

def build_index_from_bbc(df: pd.Series): # → embeddingi (np. sentence-transformers/all-MiniLM-L6-v2 lokalnie)

    if not isinstance(df, pd.Series):
        raise TypeError("df must be pd.Series")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cuda')

    all_chunks = []
    
    df = load_bbc_csv(set_type="train")
    text_list = df['text'].to_list()
    print(len(text_list))
    print(text_list[:10])
    
    all_chunks = []
    for text in text_list:
        all_chunks.extend(chunk_text(text))

    embeddings = model.encode(all_chunks)
    print(embeddings.shape)
    print(embeddings[:10])

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print(index.ntotal)
    

    k = 4
    xq = model.encode(["Someone demand a new car"])

    D, I = index.search(xq, k)  # search
    print(I)
    print(type(I))
    print(type(I[0]))
    print(I)
    print(I[0])

    print("Długość text_list:", len(text_list))
    print("Zawartość I[0]:", I[0].tolist())

    # Bezpieczne pobieranie tekstów (odrzuca indeksy -1 oraz zbyt duże)
    wyniki = [all_chunks[idx] for idx in I[0].tolist() if 0 <= idx < len(text_list)]
    print(wyniki)


    print([all_chunks[idx] for idx in I[0].tolist()])


def build_index_from_docs(docs_dir: Path = Path.cwd() / "docs"): # → embeddingi (np. sentence-transformers/all-MiniLM-L6-v2 lokalnie)

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory doesnt exist here: {docs_dir}")
    
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks = []
    
    df = load_bbc_csv(set_type="train")
    text_list = df['text'].to_list()
    print(len(text_list))
    print(text_list[:10])

    for text in text_list:
        text_chunked = chunk_text(text)
        all_chunks.extend(text_chunked)

    print(len(all_chunks))
    print(all_chunks[:10])

    # for file_path in docs_dir.iterdir():
    #     if file_path.is_file():
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             text = file.read()
    #             text_chunked = chunk_text(text)
    #             all_chunks.extend(text_chunked)

    # embeddings = model.encode(all_chunks)
    # print(embeddings.shape)
    # print(embeddings[:10])

def retrieve(query, top_k): # → cosine similarity
    pass

def answer(query): # → sklej kontekst + prompt do LLM (API lub Ollama lokalnie)
    pass

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")
    text_sample = df['text'].iloc[0]
    chunks = chunk_text(text_sample)
    print(chunks)
    print([len(x) for x in chunks])
    build_index_from_bbc(df['text'])
