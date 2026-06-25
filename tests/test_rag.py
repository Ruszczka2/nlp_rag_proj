from nlp_rag_proj.rag import chunk_text

def test_chunk_text_division():
    assert chunk_text("a "*60) == ["a "*49+"a", "a "*19+"a"]