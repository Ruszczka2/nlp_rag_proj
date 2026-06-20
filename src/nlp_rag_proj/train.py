from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from nlp_rag_proj.features import build_vectorizer
from pathlib import Path
import pandas as pd
import joblib

def train(X: pd.Series, y: pd.Series, model_path: Path = Path.cwd() / "models" / "tfidf_svc.joblib") -> Pipeline:

    if not isinstance(X, (pd.Series, pd.DataFrame)) or not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("Both args 'X' and 'y' must be type: pd.Series or pd.DataFrame")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y are diffrent sizes: X({X.shape[0]}) -> y({y.shape[0]})")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline([
        ('tfidf', build_vectorizer()),
        ('svc', LinearSVC(random_state=42))
        ])
    
    randomsearch_params = {
        'tfidf__max_features':[8000, 10000, 12000],
        #'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 2, 3],
        'svc__C': [ 1, 3, 5, 7]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = GridSearchCV(
        estimator=pipe,
        param_grid=randomsearch_params,
        cv=skf,
        n_jobs=-1,
        verbose=3,
        scoring='f1_macro'
    )

    random_search.fit(X, y)

    print("Najlepsze parametry:", random_search.best_params_)
    print("Najlepszy wynik (F1-score):", random_search.best_score_)

    joblib.dump(random_search.best_estimator_, model_path)

    return random_search.best_estimator_ 

if __name__ == "__main__":
    X = pd.Series(["aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde", "aa bb cc", "dd ee ff", "ws fde"])
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    model = train(X, y)
    for el in dir(model):
        print(el)
    print(model.decision_function())