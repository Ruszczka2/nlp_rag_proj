from pathlib import Path
import joblib
import pandas as pd

DATA_DIR = Path.cwd() / "data" / "raw"

def load_bbc_csv(*, set_type: str = None) -> pd.DataFrame:

    if not isinstance(set_type, str):
        raise TypeError(f"set_type must be {type("")}, but it is: {type(set_type)}")

    if set_type == "train":
        load_path = Path(DATA_DIR / "BBC_train.csv")
    elif set_type == "test":
        load_path = Path(DATA_DIR / "BBC_test.csv")
    else:
        raise ValueError(f"set_type can be either 'train' or 'test', but it is: {set_type}")
    
    if not load_path.exists():
        raise FileNotFoundError(f"File not found at: {load_path}")
    
    df = pd.read_csv(load_path)
    df = df.rename(columns=lambda x: x.lower())
    
    return df

def load_model(load_path: Path):

    if not load_path.is_file():
        raise FileNotFoundError(f"Model nie istnieje pod ścieżką: {load_path}")
    
    return joblib.load(load_path)

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")
    print(df.head())