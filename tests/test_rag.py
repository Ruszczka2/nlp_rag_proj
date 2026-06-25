from nlp_rag_proj.rag import chunk_text



def test_chunk_text_():
    assert chunk_text("a a a a a a a a a a a a a a a a a a a") == ['a a a a a a a a a a a a a a a a a a a', 'a a a']