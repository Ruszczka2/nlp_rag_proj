from nlp_rag_proj.clean import normalize_text

def test_normalize_text_empty_string():
    assert normalize_text("") == ""

def test_normalize_text_multiple_spaces():
    assert normalize_text("tekst    z    spacjami") == "tekst z spacjami"

def test_normalize_text_only_url():
    assert normalize_text("http://url.com") == ""

def test_normalize_text_capital_letters():
    assert normalize_text("AbcGREeŁ") == "abcgreeł"

def test_normalize_text_punctuation():
    assert normalize_text("ab, c!") == "ab c"

def test_normalize_text_whitespaces():
    assert normalize_text("ab\t\nc\n\n") == "ab c"

def test_normalize_text_numbers():
    assert normalize_text("ab15 c8888") == "ab c"