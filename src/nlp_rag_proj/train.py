from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from nlp_rag_proj.features import build_vectorizer
from nlp_rag_proj.predict import load_model
from pathlib import Path
import pandas as pd
import joblib

def build_pipe(*, linear_svc: bool = True):

    if linear_svc:
        svc = LinearSVC(random_state=42)

        gridsearch_params = {
            'tfidf__max_features':[8000, 10000, 12000],
            'tfidf__min_df': [1, 2, 3],
            'svc__C': [ 1, 3, 5, 7]
        }

    else:
        svc = SVC(random_state=42)

        gridsearch_params = [{
            'tfidf__max_features':[8000, 10000, 12000],
            'tfidf__min_df': [1, 2],
            'svc__kernel': ['rbf'], 
            'svc__decision_function_shape': ['ovr', 'ovo'],
            'svc__C': [1, 10, 20], 
        },
        {
            'tfidf__max_features':[8000, 10000, 12000],
            'tfidf__min_df': [1, 2],
            'svc__kernel': ['poly'], 
            'svc__degree': [2],
            'svc__decision_function_shape': ['ovr', 'ovo'],
            'svc__C': [1, 10, 20], 
        }]

    pipe = Pipeline([
        ('tfidf', build_vectorizer()),
        ('svc', svc)
        ])

    return pipe, gridsearch_params

def train_model(pipe: Pipeline, gridsearch_params: dict, verbose: int = 3):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=gridsearch_params,
        cv=skf,
        n_jobs=-1,
        verbose=verbose,
        scoring='f1_macro'
    )

    return grid_search

def train(X: pd.Series, y: pd.Series, model_path: Path, *, verbose: int = 3, linear_svc: bool = True) -> Pipeline:

    if not isinstance(X, (pd.Series, pd.DataFrame)) or not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("Both args 'X' and 'y' must be type: pd.Series or pd.DataFrame")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y are diffrent sizes: X({X.shape[0]}) -> y({y.shape[0]})")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)

    pipe, gridsearch_params = build_pipe(linear_svc=linear_svc)

    grid_search = train_model(pipe, gridsearch_params, verbose)

    grid_search.fit(X, y)

    print("Best Params:", grid_search.best_params_)
    print("Best Score (F1-score):", grid_search.best_score_)

    joblib.dump(grid_search.best_estimator_, model_path)

    return grid_search.best_estimator_ 

if __name__ == "__main__":
    model = load_model(Path.cwd() / "models" / "tfidf_svc_nl.joblib")
    for el in model.get_params():
        print(el)
