import json
import re
import emoji

def load_json_array(path: str) -> list:
    """
    Надёжно читает любой большой JSON-массив из одного файла,
    вычленяя объекты {…} по учёту вложенности скобок.
    """
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

def clean_text(text: str) -> str:
    """
    Убирает эмодзи, ссылки, Telegram-разметку, хэштеги и спецсимволы.
    Оставляет только буквы (рус+лат) и цифры.
    """
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\.\S+|https\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)      # [club123|@foo]
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^\w\sа-яА-ЯёЁ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
