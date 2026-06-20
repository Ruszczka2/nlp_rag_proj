from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path
import pandas as pd
import joblib



def load_model(load_path: Path = Path(Path.cwd() / "models" / "tfidf_svc.joblib")):

    if not load_path.is_file():
        raise FileNotFoundError(f"Model nie istnieje pod ścieżką: {load_path}")
    
    return joblib.load(load_path)


def predict_category(
        X_test: pd.Series,
        model: Pipeline
    ) -> str:

    return model.predict(X_test)

def evaluate_predictions(
        y_pred: str,
        y_test: pd.Series
    ) -> None:

    print("\nRaport Klasyfikacji:")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    load_model()