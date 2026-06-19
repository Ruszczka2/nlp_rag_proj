import joblib
from pathlib import Path


def load_model(load_path: Path):

    if not load_path.is_file():
        raise FileNotFoundError(f"Model nie istnieje pod ścieżką: {load_path}")
    
    return joblib.load(load_path)


def predict_category(text) -> str:
    pass


if __name__ == "__main__":
    load_model()