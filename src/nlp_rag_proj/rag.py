from nlp_rag_proj.io import load_bbc_csv
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import faiss

def embed_query(queries: list[str], emb: str = 'docs') -> np.ndarray:
    if emb == 'docs':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1", device='cuda')
    elif emb == 'bbc':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cuda')
    else:
        raise ValueError("emb must be either 'docs' or 'bbc'")
    
    return model.encode(queries)

def calc_cosine_similarity(queries: list[str], emb: str = 'docs') -> np.ndarray[np.float32]:

    if emb == 'docs':
        with open(file=Path.cwd() / "data" / "emb" / "emb_dict_docs.pickle", mode='rb') as f:
            emb_dict = pickle.load(f, encoding='utf-8')
    elif emb == 'bbc':
        with open(file=Path.cwd() / "data" / "emb" / "emb_dict.pickle", mode='rb') as f:
            emb_dict = pickle.load(f, encoding='utf-8')
    else:
        raise ValueError("emb must be either 'docs' or 'bbc'")

    queries_vect = embed_query(queries, emb=emb)

    embeddings_vect = np.array(emb_dict['embeddings'])

    # Znormalizowane do 1 (L2) - nie muszę dzielić przez pierwiastki kwadratów v1 i v2 - wystarczy mnożenie macierzowe
    similar_array = np.dot(embeddings_vect, queries_vect.T)

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
    
    text_normalized = text #normalize_text(text, rag=True)
    text_list = text_normalized.split()

    step = chunk_size - overlap
    chunks = [' '.join(text_list[i:i+chunk_size]) for i in range(0, len(text_list), step)]
    return chunks

def build_index_from_bbc() -> faiss.IndexFlatL2: # → embeddingi
    
    df = load_bbc_csv(set_type="train")
    text_list = df['text'].to_list()
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cuda')
    
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

def build_index_from_docs(docs_dir: Path = Path.cwd() / "docs") -> faiss.IndexFlatL2: # → embeddingi

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory doesnt exist here: {docs_dir}")
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

    all_chunks = []

    for file_path in docs_dir.iterdir():
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                all_chunks.extend(chunk_text(text))

    embeddings = model.encode(all_chunks)

    # Tu jest automatyczne wykorzystanie biblioteki faiss
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    # index.add(embeddings)

    print(all_chunks)

    output_file = Path.cwd() / "data" / "emb" / "emb_dict_docs.pickle"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump({'chunks': all_chunks,'embeddings': embeddings}, f)
    
    return {'chunks': all_chunks,'embeddings': embeddings}

def retrieve(queries: list[str], top_k: int = 4, emb: str = 'docs'): # → cosine similarity
    # xq = model.encode([query])
    # _, I = index.search(xq, top_k)
    # return(I)

    if emb == 'docs':
        with open(file=Path.cwd() / "data" / "emb" / "emb_dict_docs.pickle", mode='rb') as f:
            emb_dict = pickle.load(f, encoding='utf-8')
    elif emb == 'bbc':
        with open(file=Path.cwd() / "data" / "emb" / "emb_dict.pickle", mode='rb') as f:
            emb_dict = pickle.load(f, encoding='utf-8')
    else:
        raise ValueError("emb must be either 'docs' or 'bbc'")

    similar_array = calc_cosine_similarity(queries, emb)

    idx_top_k = []

    for i in range(similar_array.shape[1] if similar_array.ndim > 1 else 1):
        query_sim = similar_array[:, i] if similar_array.ndim > 1 else similar_array
        # '-query_sim' - żeby malejąco
        sorted_indices = np.argsort(-query_sim)
        
        if top_k > len(sorted_indices):
            raise ValueError(f"top_k ({top_k}) must be lower than {len(sorted_indices)}")
            
        idx_top_k.extend(sorted_indices[:top_k])

    # Usunięcie duplikatów przy zachowaniu kolejności
    idx_top_k = list(dict.fromkeys(idx_top_k))

    return [emb_dict['chunks'][i] for i in idx_top_k]

def answer(queries: list[str], top_k: int = 4, emb: str = 'docs'):
    from ollama import chat

    if emb not in ['docs', 'bbc']:
        raise ValueError("emb must be either 'docs' or 'bbc'")

    relevant_chunks = retrieve(queries, top_k=top_k, emb=emb)
    
    context = "\n---\n".join(relevant_chunks)

    system_instruction = (
        "Jesteś pomocnym asystentem. Odpowiedz na pytanie użytkownika wyłącznie "
        "na podstawie dostarczonego kontekstu. Jeśli w tekście nie ma odpowiedzi, "
        "napisz bezpośrednio, że nie dysponujesz taką wiedzą."
    )
    answers = []
    for query in queries:
        user_content = f"Kontekst:\n{context}\n\nPytanie: {query}"

        response = chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_content}
            ],
        )
        answers.append(response.message.content)
        
    return answers

if __name__ == "__main__":

    # build_index_from_docs()
    # build_index_from_bbc()
    pass
