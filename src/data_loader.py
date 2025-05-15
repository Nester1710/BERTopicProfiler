import csv
from pathlib import Path

def load_raw_records(raw_dir: str) -> list[dict]:
    records = []
    for csv_path in Path(raw_dir).glob('*.csv'):
        try:
            with csv_path.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({
                        'dataset': row.get('collection', '').strip() or None,
                        'message_id': int(row['message_id']) if row.get('message_id') else None,
                        'text': row.get('text', '').strip(),
                        'date': float(row['date']) if row.get('date') else None
                    })
        except Exception as e:
            print(f"Warning: не удалось загрузить {csv_path.name}: {e}")
    return records
