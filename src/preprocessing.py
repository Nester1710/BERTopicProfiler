import re
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

STOP_PHRASES = ['ПОДПИСАТЬСЯ на АГП ЯЛТА КРЫМ']

def preprocess_text(text: str) -> str:
    # Удаляем URL
    text = re.sub(r'http\S+|www\.]\S+', '', text)
    # Удаляем хэштеги
    text = re.sub(r'#\w+', '', text)
    # Удаляем телефонные номера (+1234567890, 12-34-56 и т.п.)
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', '', text)
    # Оставляем только буквы, цифры и пробелы
    text = re.sub(r'[^0-9A-Za-zА-Яа-яЁё\s]', ' ', text)
    # Сжимаем несколько пробелов в один
    text = re.sub(r'\s+', ' ', text).strip()

    # Удаляем точные фразы из STOP_PHRASES
    for phrase in STOP_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Сжимаем пробелы после удаления фраз
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _process_record(record: dict) -> dict:
    original = record.get('text', '') or ''
    cleaned = preprocess_text(original)
    count = len(cleaned.split())
    return {
        'original_text': original,
        'cleaned_text': cleaned,
        'word_count': count
    }

def preprocess_records(records: list) -> pd.DataFrame:
    total = len(records)
    with Pool(cpu_count()) as pool:
        processed = list(tqdm(pool.imap(_process_record, records), total=total))
    df = pd.DataFrame(processed, columns=['original_text', 'cleaned_text', 'word_count'])
    return df
