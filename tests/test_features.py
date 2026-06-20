from nlp_rag_proj.features import build_vectorizer, prepare_dataset
from nlp_rag_proj.clean import normalize_text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
import pytest

@pytest.fixture
def dummy_dataframe():
    # Definiujemy bazowe dane (10 wierszy), w których każda klasa ma min. 2 wpisy
    base_data = {
        "Text": [
            "Article one text", "Article two text", 
            "Article three text", "Article four text", 
            "Article five text", "Article six text",
            "Article seven text", "Article eight text",
            "Article nine text", "Article ten text"
        ],
        "Category": [
            "tech", "tech", 
            "business", "business", 
            "sport", "sport",
            "politics", "politics",
            "entertainment", "entertainment"
        ]
    }
    df_base = pd.DataFrame(base_data)
    
    df_large = pd.concat([df_base, df_base, df_base, df_base, df_base], ignore_index=True)
    return df_large



def test_prepare_dataset_correct_length(dummy_dataframe):
    assert len(prepare_dataset(dummy_dataframe)) == 4

def test_prepare_dataset_correct_datatypes(dummy_dataframe):
    result = prepare_dataset(dummy_dataframe)
    
    assert isinstance(result, tuple)
    assert len(result) == 4
    
    for item in result:
        assert isinstance(item, pd.Series)

def test_prepare_dataset_retains_unique_classes_in_splits(dummy_dataframe):
    _, _, y_train, y_test = prepare_dataset(dummy_dataframe)
    correct_category_length = len(pd.unique(dummy_dataframe["Category"]))

    assert len(pd.unique(y_train)) == correct_category_length
    assert len(pd.unique(y_test)) == correct_category_length



def test_build_vectorizer_correct_datatype():
    vectorizer = build_vectorizer()
    assert isinstance(vectorizer, TfidfVectorizer)

def test_build_vectorizer_params():
    vectorizer = build_vectorizer()
    assert vectorizer.get_params()["preprocessor"] == normalize_text
    assert vectorizer.get_params()["stop_words"] == list(ENGLISH_STOP_WORDS.union({"s", "-", "said", "new"}))
