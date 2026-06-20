from nlp_rag_proj.clean import normalize_text

def test_normalize_text_no_string():
    assert normalize_text([]) == ""

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

def test_normalize_text_complex():
    assert normalize_text("Check this 123 and pay $10! https://www.google.com He said it is a new project.") == "check this and pay $ he said it is a new project"

def test_normalize_text_already_normalized():
    assert normalize_text("check this he said it is a new project") == "check this he said it is a new project"