import re
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import unicodedata

STOP_PHRASES = []

def preprocess_text(text: str) -> str:

    # Приводим текст к нижнему регистру
    text = unicodedata.normalize('NFKC', text).lower()
    # Удаляем URL
    re.sub(r'https?://\S+|www\.\S+', '', text)
    # Удаляем хэштеги
    text = re.sub(r'#\w+', '', text)
    # Удаляем телефонные номера (+1234567890, 12-34-56 и т.п.)
    re.sub(r'(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{2}[\s-]?\d{2}', '', text)
    # Оставляем только буквы, цифры и пробелы
    text = re.sub(r'[^0-9A-Za-zА-Яа-яЁё\s]', ' ', text)
    # Сжимаем несколько пробелов в один
    text = re.sub(r'\s+', ' ', text).strip()

    # Удаляем точные фразы из STOP_PHRASES
    for phrase in STOP_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Сжимаем пробелы после удаления фраз
    return re.sub(r'\s+', ' ', text).strip()

def _process_record(args: tuple) -> dict:
    idx, record = args
    original = record.get('text', '') or ''
    cleaned = preprocess_text(original)
    count = len(cleaned.split())
    dataset = record.get('dataset', '')
    return {
        'id': idx,
        'dataset': dataset,
        'original_text': original,
        'cleaned_text': cleaned,
        'word_count': count
    }

def preprocess_records(records: list) -> pd.DataFrame:
    total = len(records)
    # Подготовка аргументов для параллельной обработки: кортеж (id, record)
    args_list = [(i, rec) for i, rec in enumerate(records, start=1)]

    with Pool(cpu_count()) as pool:
        processed = list(tqdm(pool.imap(_process_record, args_list), total=total, desc='Preprocessing'))

    df = pd.DataFrame(processed, columns=['id', 'dataset', 'original_text', 'cleaned_text', 'word_count'])
    return df
