import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from nlp_rag_proj.train import build_pipe, train

@pytest.fixture
def model_test_path(tmp_path):
    return tmp_path / "test.joblib"

@pytest.fixture
def dummy_dataframe():
    X = pd.Series(["aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde"])
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    X_test = X.iloc[::-1].reset_index(drop=True)
    y_test = y.iloc[::-1].reset_index(drop=True)
    df = pd.DataFrame({'X': X, 'y': y, 'X_test': X_test, 'y_test': y_test})
    return df

def test_build_pipe_params():
    pipe_linear, params_linear = build_pipe(linear_svc=True)
    assert isinstance(pipe_linear, Pipeline)
    assert isinstance(params_linear, dict)
    assert 'svc__C' in params_linear
    assert 'tfidf__max_features' in params_linear

    pipe_nonlinear, params_nonlinear = build_pipe(linear_svc=False)
    assert isinstance(pipe_nonlinear, Pipeline)
    assert isinstance(params_nonlinear, list)
    assert len(params_nonlinear) == 2

def test_train_decision(dummy_dataframe, model_test_path):
    X = dummy_dataframe["X"]
    y = dummy_dataframe["y"]
    
    model = train(X, y, model_path=model_test_path, verbose=0)
    
    assert model_test_path.exists()
    
    X_test = dummy_dataframe["X_test"]
    y_test = dummy_dataframe["y_test"]
    
    decision_scores = model.decision_function(X_test)
    assert decision_scores.shape == (len(X_test), 3)
    
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)