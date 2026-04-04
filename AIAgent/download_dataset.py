"""
Скрипт для скачивания датасета retail-dataset из Kaggle.
"""

import kagglehub
import pandas as pd
import os

def download_retail_dataset():
    """Скачивает и объединяет датасет retail-dataset."""
    path = kagglehub.dataset_download('ebruiserisobay/retail-dataset')
    print(f"📁 Датасет загружен: {path}")

    # Предполагаем, что в папке есть retail.csv или аналог
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        print(f"✅ Загружено {len(df)} записей")
        # Сохранить в data/retail_full.csv
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, 'retail_full.csv')
        df.to_csv(output_path, index=False)
        print(f"💾 Сохранено в {output_path}")
    else:
        print("❌ CSV файлы не найдены")

if __name__ == "__main__":
    download_retail_dataset()