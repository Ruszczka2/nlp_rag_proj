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
        raise FileNotFoundError(f"Model doesnt exist here: {load_path}")
    
    return joblib.load(load_path)

def extract_random_articles(df: pd.DataFrame, num_articles: int, ext_path: Path) -> None:
    
    if not isinstance(num_articles, int):
        raise TypeError(f"num_articles must be {type(int)}, but it is: {type(num_articles)}")
    
    ext_path.mkdir(parents=True, exist_ok=True)
    
    df_sampled = df.sample(n=num_articles, random_state=42)

    for index, row in df_sampled.iterrows():
        file_name = f"file{index}_{row["category"]}.txt"
        
        text_to_save = str(row['text'])
        
        with open(ext_path / file_name, "w", encoding="utf-8") as plik:
            plik.write(text_to_save)

if __name__ == "__main__":
    df = load_bbc_csv(set_type="train")
    print(df.head())