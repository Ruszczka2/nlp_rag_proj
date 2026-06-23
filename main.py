from pathlib import Path
from typing import Any
import pandas as pd
import argparse
import time

from nlp_rag_proj import clean, features, io, parrallel, predict, rag, sample, tokenization, train

def multiprocess_test(args: Any | None = None) -> None:
    
    df = io.load_bbc_csv(set_type="train")

    extracted_series = parrallel.extract_random_articles(num_articles=args.articles_num, df=df)

    start_time = time.perf_counter()
    parrallel.sequential(extracted_series)
    end_time = time.perf_counter()

    sequential_elapsed_time = end_time - start_time
    print(f"Operation Sequential: {sequential_elapsed_time:.6f} sec")

    start_time = time.perf_counter()
    parrallel.parrallel(extracted_series)
    end_time = time.perf_counter()

    sequential_elapsed_time = end_time - start_time
    print(f"Operation Sequential: {sequential_elapsed_time:.6f} sec")







def tfidf_svc_pipeline(args: Any | None = None, model_path: Path = Path.cwd() / "models" / "tfidf_svc.joblib") -> None:

    df = io.load_bbc_csv(set_type="train")

    X_train, X_test, y_train, y_test = features.prepare_dataset(df, split_test=True)

    stemmer = None
    nlp_obj = None
    path_suf = None
    linear_svc = True

    if args.stem:
        import Stemmer
        print("\nLoading stem model...")
        stemmer = Stemmer.Stemmer('english')
        path_suf = "stem_tfidf_svc"

    elif args.lem:
        import spacy
        tokenization.init_gpu()
        print("\nLoading lemmatization model...")
        nlp_obj = spacy.load("en_core_web_trf")
        path_suf = "lem_tfidf_svc"

    else:
        path_suf = "tfidf_svc"

    if args.non_linear:
        path_suf += '_nl'
        linear_svc = False

    model_path = Path.cwd() / "models" / f"{path_suf}.joblib"

    X_test = clean.apply_nlp(X_test, args, stemmer=stemmer, nlp_obj=nlp_obj)

    if model_path.exists() and not args.force_train:
        try:
            model = io.load_model(model_path)
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Error during model loading from {model_path}: {e}\nTraining model...\n")
            X_train = clean.apply_nlp(X_train, args, stemmer=stemmer, nlp_obj=nlp_obj)
            model = train.train(X=X_train, y=y_train, model_path=model_path, linear_svc=linear_svc)
    else:
        if not model_path.exists():
            print(f"Model has not beed found: {model_path}\nTraining model...\n")
        elif args.force_train:
            print(f"Force train flag is: {args.force_train}\nTraining model...\n")

        X_train = clean.apply_nlp(X_train, args, stemmer=stemmer, nlp_obj=nlp_obj)
        model = train.train(X=X_train, y=y_train, model_path=model_path, linear_svc=linear_svc)

    y_pred = predict.predict_category(X_test=X_test, model=model)

    predict.evaluate_predictions(y_pred, y_test, model=model)


    # --- Utworzenie submission na platforme Kaggle i sprawdzenie wyników ---
    if args.submit:
        df_test = io.load_bbc_csv(set_type="test")

        article_ids = df_test["articleid"]
        X_test_test = clean.apply_nlp(df_test["text"], args, stemmer=stemmer, nlp_obj=nlp_obj)

        y_test_pred = model.predict(X_test_test)

        submission_df = pd.DataFrame({
            'ArticleId': article_ids,
            'Category': y_test_pred
        })

        submission_df.to_csv(Path.cwd() / "outputs" / f"{path_suf}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Wygeneruj plik submission_kaggle.csv")
    parser.add_argument("-ft", "--force-train", action="store_true", help="Wymuś trenowanie modelu mimo już istniejącego")
    parser.add_argument("-nl", "--non-linear", action="store_true", help="Użycie nieliniowego kernela dla klasyfikatora SVM")
    parser.add_argument("-an", "--articles_num", type=int, default=None, help="Użycie nieliniowego kernela dla klasyfikatora SVM")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", "--no-norm", action="store_true", help="Nie normalizuj tekstu")
    group.add_argument("-s", "--stem", action="store_true", help="Stemming tekstu")
    group.add_argument("-l", "--lem", action="store_true", help="Lematyzacja tekstu")
    args = parser.parse_args()

    if args.articles_num is None:
        tfidf_svc_pipeline(args)

    else:
        if args.articles_num < 100:
            print(f"min articles is 100, but was given {args.articles_num}. Changing to 100")
            args.articles_num = 100

        if args.articles_num > 500:
            print(f"max articles is 500, but was given {args.articles_num}. Changing to 500")
            args.articles_num = 500
