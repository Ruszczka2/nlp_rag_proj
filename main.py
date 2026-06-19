import nlp_rag_proj as nlp
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import pandas as pd


def main():
    # tablica = np.array([[0,1,2],[5,8,7]], dtype=np.int8)
    # print(tablica.shape)
    df = nlp.io.load_bbc_csv(set_type="train")
    print(df.head())

    X_train, X_test, y_train, y_test = nlp.features.prepare_dataset(df)
    for pds in [X_train, X_test, y_train, y_test]:
        print(pds.head())

    model = joblib.load(Path.cwd() / "models" / "tfidf_svc.joblib")

    y_pred = model.predict(X_test)

    print("Raport Klasyfikacji:")
    print(classification_report(y_test, y_pred))

    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))

    df_test = nlp.io.load_bbc_csv(set_type="test")

    article_ids = df_test["ArticleId"]
    X_test_test = df_test["Text"]

    y_test_pred = model.predict(X_test_test)

    submission_df = pd.DataFrame({
        'ArticleId': article_ids,
        'Category': y_test_pred
    })

    submission_df.to_csv('submission_kaggle.csv', index=False)

if __name__ == "__main__":
    main()
