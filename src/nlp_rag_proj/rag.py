from nlp_rag_proj.clean import normalize_text
from nlp_rag_proj.io import load_bbc_csv
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import faiss

def embed_query(query: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cuda')
    return model.encode(query)

def calc_cosine_similarity(query: str) -> np.ndarray[np.float32]:

    with open(file=Path.cwd() / "data" / "emb" / "emb_dict.pickle", mode='rb') as f:
        emb_dict = pickle.load(f, encoding='utf-8')

    query_vect = embed_query(query)

    embeddings_vect = np.array(emb_dict['embeddings'])

    # Znormalizowane do 1 (L2) - nie muszę dzielić przez pierwiastki kwadratów v1 i v2 - wystarczy mnożenie macierzowe
    similar_array = np.dot(embeddings_vect, query_vect)

    #sorted_similar_array = similar_array[sorted_indices]
    #id_sim = (sorted_idx, sorted_similar_array)
    
    return similar_array

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

def build_index_from_bbc(df: pd.Series) -> faiss.IndexFlatL2: # → embeddingi

    if not isinstance(df, pd.Series):
        raise TypeError("df must be pd.Series")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cuda')

    text_list = df.to_list()
    
    all_chunks = []
    for text in text_list:
        all_chunks.extend(chunk_text(text))

    embeddings = model.encode(all_chunks)

    # Tu jest automatyczne wykorzystanie biblioteki faiss
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    # index.add(embeddings)

    output_file = Path.cwd() / "data" / "emb" / "emb_dict.pickle"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump({'chunks': all_chunks,'embeddings': embeddings}, f)
    
    return {'chunks': all_chunks,'embeddings': embeddings}

def build_index_from_docs(docs_dir: Path = Path.cwd() / "docs"): # → embeddingi

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory doesnt exist here: {docs_dir}")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

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

    for file_path in docs_dir.iterdir():
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text_chunked = chunk_text(text)
                all_chunks.extend(text_chunked)

    embeddings = model.encode(all_chunks)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index

def retrieve(query: str, top_k: int = 4): # → cosine similarity
    # xq = model.encode([query])
    # _, I = index.search(xq, top_k)
    # return(I)

    similar_array = calc_cosine_similarity(query)

    idx = np.arange(similar_array.shape[0])
    # '-similar_array' - żeby malejąco
    sorted_indices = np.argsort(-similar_array)
    sorted_idx = idx[sorted_indices]

    if top_k > sorted_idx.shape[0]:
        raise ValueError(f"top_k ({top_k}) must be lower than {sorted_idx.shape[0]}")

    idx_top_k = sorted_idx[:top_k]
    print(idx_top_k)

def answer(query): # → sklej kontekst + prompt do LLM (API lub Ollama lokalnie)
    pass

if __name__ == "__main__":
    with open(file=Path.cwd() / "data" / "emb" / "emb_dict.pickle", mode='rb') as f:
        emb_dict = pickle.load(f, encoding='utf-8')
    #retrieve('a person who is president of USA.', 5)

    calc_cosine_similarity(('a person who is president of USA.'))

    

