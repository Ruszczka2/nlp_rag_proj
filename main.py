import nlp_rag_proj as nlp
from pathlib import Path
import pandas as pd


def main():

    df = nlp.io.load_bbc_csv(set_type="train")
    model_path = Path.cwd() / "models" / "tfidf_svc.joblib"

    X_train, X_test, y_train, y_test = nlp.features.prepare_dataset(df, True)

    if model_path.exists():
        model = nlp.predict.load_model()
    else:
        print(f"Nie znaleziono modelu w: {model_path}\n\nTrenuje model...\n")
        model = nlp.train.train(X_train, y_train, model_path)

    y_pred = nlp.predict.predict_category(X_test, model)

    nlp.predict.evaluate_predictions(y_pred, y_test)


    # --- Utworzenie submission na platforme Kaggle i sprawdzenie wyników ---
    df_test = nlp.io.load_bbc_csv(set_type="test")

    article_ids = df_test["articleid"]
    X_test_test = df_test["text"]

    y_test_pred = model.predict(X_test_test)

    submission_df = pd.DataFrame({
        'ArticleId': article_ids,
        'Category': y_test_pred
    })

    submission_df.to_csv('submission_kaggle.csv', index=False)

if __name__ == "__main__":
    main()
