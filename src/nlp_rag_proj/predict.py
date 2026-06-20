from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from typing import Literal

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def load_model(load_path: Path = Path(Path.cwd() / "models" / "tfidf_svc.joblib")):

    if not load_path.is_file():
        raise FileNotFoundError(f"Model nie istnieje pod ścieżką: {load_path}")
    
    return joblib.load(load_path)


def predict_category(
        X_test: pd.Series,
        model: Pipeline
    ) -> np.ndarray[tuple[int, int], np.dtype[str]]:

    print(type(model.predict(X_test)))

    return model.predict(X_test)

def evaluate_predictions(
        y_pred: np.ndarray[tuple[int, int], np.dtype[str]],
        y_test: pd.Series,
        model: Pipeline | None = None
    ) -> None:

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if model:
        classes = model.named_steps['svc'].classes_
        cm = confusion_matrix(y_test, y_pred, labels=classes)
    else:
        classes = range(0,5)
        cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.rcParams.update({'font.size': 18})
    sns.heatmap(cm, fmt='d', cmap='Blues', annot=True, xticklabels=classes, yticklabels=classes)
    plt.title('Macierz pomyłek (Confusion Matrix)', pad=20) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



if __name__ == "__main__":
    y_pred = ""
    load_model()