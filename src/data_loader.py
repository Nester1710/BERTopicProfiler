import os
import json

def load_json_array(path: str) -> list:
    text = open(path, "r", encoding="utf-8").read()
    start = text.find("[")
    end   = text.rfind("]")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Нет JSON-массива в квадратных скобках")
    inner = text[start+1:end]

    records = []
    depth = 0
    obj_start = None
    for i, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_str = inner[obj_start : i+1]
                try:
                    records.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return records

def load_raw_records(raw_dir: str) -> list:
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
