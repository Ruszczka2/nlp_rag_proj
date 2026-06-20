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

    X_train, X_test, y_train, y_test = nlp.features.prepare_dataset(df, True)

    if args.stem:
        print("\nLoading stem model...\n")
        stemmer = Stemmer.Stemmer('english')
        text_normalization = partial(nlp.tokenization.stem, nlp_object=stemmer)
    if args.lem:
        print("\nLoading lemmatization model...\n")
        nlp_obj = spacy.load("en_core_news_trf")
        text_normalization = partial(nlp.tokenization.lemmatize, nlp_object=nlp_obj)
        return 0
    else:
        text_normalization = nlp.clean.normalize_text

    if model_path.exists():
        model = nlp.predict.load_model()
    else:
        print(f"Nie znaleziono modelu w: {model_path}\n\nTrenuje model...\n")
        model = nlp.train.train(X=X_train, y=y_train, model_path=model_path, text_normalization=text_normalization)

    y_pred = nlp.predict.predict_category(X_test=X_test, model=model)

    nlp.predict.evaluate_predictions(y_pred, y_test, model=model)


    # --- Utworzenie submission na platforme Kaggle i sprawdzenie wyników ---
    if args.submit:
        df_test = nlp.io.load_bbc_csv(set_type="test")

        article_ids = df_test["articleid"]
        X_test_test = df_test["text"]

        y_test_pred = model.predict(X_test_test)

        submission_df = pd.DataFrame({
            'ArticleId': article_ids,
            'Category': y_test_pred
        })

        submission_df.to_csv(Path.cwd() / "outputs" / "submission_kaggle.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--submit", action="store_true", help="Wygeneruj plik submission_kaggle.csv")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--no-norm", action="store_true", help="Nie normalizuj tekstu")
    group.add_argument("-st", "--stem", action="store_true", help="Stemming tekstu")
    group.add_argument("-l", "--lem", action="store_true", help="Lematyzacja tekstu")
    args = parser.parse_args()

    tfidf_svc_pipeline(args)
