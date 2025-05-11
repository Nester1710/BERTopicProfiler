import os
from .utils import load_json_array

def load_raw_records(raw_dir: str) -> list:
    """
    Обходит все .json в папке raw_dir и возвращает единый список словарей.
    """
    records = []
    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(raw_dir, fname)
        try:
            recs = load_json_array(path)
            records.extend(recs)
        except Exception as e:
            print(f"Warning: не удалось загрузить {fname}: {e}")
    return records
