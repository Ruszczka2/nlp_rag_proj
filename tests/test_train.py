from nlp_rag_proj.train import build_pipe, train, train_model
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from pathlib import Path
import pandas as pd
import pytest

@pytest.fixture
def model_test_path():
    return Path.cwd() / "models" / "test.joblib"

@pytest.fixture
def dummy_dataframe():
    X = pd.Series(["aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde"])
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    X_test = X.iloc[::-1]
    y_test = y.iloc[::-1]
    df = pd.DataFrame({'X': X, 'y': y, 'X_test': X_test, 'y_test': y_test})
    return df

def test_build_pipe_params():
    pipe = build_pipe()
    assert True


def test_train_(dummy_dataframe):
    X = dummy_dataframe["X"]
    y = dummy_dataframe["y"]
    model = train(X, y, model_path = Path.cwd() / "models" / "test.joblib", verbose = 3)
    for el in dir(model):
        print(el)
    X_test = dummy_dataframe["X_test"]
    y_test = dummy_dataframe["y_test"]
    print(model.decision_function(X_test))
    print(y_test)
    assert True
