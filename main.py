from pathlib import Path
from typing import Any
import pandas as pd
import argparse

import spacy
from functools import partial
import Stemmer

import nlp_rag_proj as nlp


def tfidf_svc_pipeline(args: Any | None = None, model_path: Path = Path.cwd() / "models" / "tfidf_svc.joblib"):

    df = nlp.io.load_bbc_csv(set_type="train")

    X_train, X_test, y_train, y_test = nlp.features.prepare_dataset(df, split_test=True)

    stemmer = None
    nlp_obj = None
    path_suf = None

    if args.stem:
        print("\nLoading stem model...")
        stemmer = Stemmer.Stemmer('english')
        path_suf = "stem_tfidf_svc"

    elif args.lem:
        nlp.tokenization.init_gpu()
        print("\nLoading lemmatization model...")
        nlp_obj = spacy.load("en_core_web_trf")
        path_suf = "lem_tfidf_svc"

    else:
        path_suf = "tfidf_svc"

    model_path = Path.cwd() / "models" / f"{path_suf}.joblib"

    X_test = nlp.clean.apply_nlp(X_test, args, stemmer=stemmer, nlp_obj=nlp_obj)

    if model_path.exists():
        model = nlp.predict.load_model(model_path)
    else:
        X_train = nlp.clean.apply_nlp(X_train, args, stemmer=stemmer, nlp_obj=nlp_obj)
        print(f"Model has not beed found: {model_path}\nTraining model...\n")
        model = nlp.train.train(X=X_train, y=y_train, model_path=model_path)

    y_pred = nlp.predict.predict_category(X_test=X_test, model=model)

    nlp.predict.evaluate_predictions(y_pred, y_test, model=model)


    # --- Utworzenie submission na platforme Kaggle i sprawdzenie wyników ---
    if args.submit:
        df_test = nlp.io.load_bbc_csv(set_type="test")

        article_ids = df_test["articleid"]
        X_test_test = nlp.clean.apply_nlp(df_test["text"], args, stemmer=stemmer, nlp_obj=nlp_obj)

        y_test_pred = model.predict(X_test_test)

        submission_df = pd.DataFrame({
            'ArticleId': article_ids,
            'Category': y_test_pred
        })

        submission_df.to_csv(Path.cwd() / "outputs" / f"{path_suf}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Wygeneruj plik submission_kaggle.csv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--no-norm", action="store_true", help="Nie normalizuj tekstu")
    group.add_argument("-s", "--stem", action="store_true", help="Stemming tekstu")
    group.add_argument("-l", "--lem", action="store_true", help="Lematyzacja tekstu")
    args = parser.parse_args()

    tfidf_svc_pipeline(args)
