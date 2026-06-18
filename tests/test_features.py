from nlp_rag_proj.features import build_vectorizer, prepare_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
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
            "tech", "business",
            "tech", "sport"
        ]
    }
    df_base = pd.DataFrame(base_data)
    
    # Podwajamy zbiór danych (do 20 wierszy), aby test_size przy 20% wynosił 4 elementy
    df_large = pd.concat([df_base, df_base], ignore_index=True)
    return df_large

def test_build_vectorizer_correct_datatype():
    vectorizer = build_vectorizer()
    assert isinstance(vectorizer, TfidfVectorizer)

def test_prepare_dataset_correct_length(dummy_dataframe):
    assert len(prepare_dataset(dummy_dataframe)) == 4

def test_prepare_dataset_correct_datatypes(dummy_dataframe):
    result = prepare_dataset(dummy_dataframe)
    
    # Sprawdzamy, czy funkcja zwróciła dokładnie 4 elementy
    assert isinstance(result, tuple)
    assert len(result) == 4
    
    # Sprawdzamy, czy każdy element jest Serią danych Pandas
    for item in result:
        assert isinstance(item, pd.Series)