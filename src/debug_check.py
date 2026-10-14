import os
import pandas as pd
from src.config import EMBEDDINGS_DIR, RAW_DATA_DIR


def check_files():
    """Проверяем наличие необходимых файлов"""
    print("=== ПРОВЕРКА ФАЙЛОВ ===")

    # Проверяем структуру папок
    folders = {
        'RAW_DATA_DIR': RAW_DATA_DIR,
        'EMBEDDINGS_DIR': EMBEDDINGS_DIR
    }

    for name, path in folders.items():
        print(f"{name}: {path}")
        print(f"  Существует: {os.path.exists(path)}")
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"  Файлы: {files}")
        print()

    # Проверяем основные файлы
    files_to_check = [
        os.path.join(RAW_DATA_DIR, 'nodes.csv'),
        os.path.join(RAW_DATA_DIR, 'edges.csv'),
        os.path.join(EMBEDDINGS_DIR, 'exp_001_emb.csv')
    ]

    print("=== ПРОВЕРКА КЛЮЧЕВЫХ ФАЙЛОВ ===")
    for file_path in files_to_check:
        print(f"Файл: {file_path}")
        exists = os.path.exists(file_path)
        print(f"  Существует: {exists}")

        if exists:
            try:
                df = pd.read_csv(file_path)
                print(f"  Размер: {df.shape}")
                if 'index_col=0' in file_path or 'emb.csv' in file_path:
                    df = pd.read_csv(file_path, index_col=0)
                    print(f"  Первые индексы: {list(df.index[:5])}")
                else:
                    print(f"  Первые строки:")
                    print(df.head(2))
            except Exception as e:
                print(f"  Ошибка чтения: {e}")
        print()


def check_graph_data():
    """Анализируем данные графа"""
    nodes_path = os.path.join(RAW_DATA_DIR, 'nodes.csv')
    edges_path = os.path.join(RAW_DATA_DIR, 'edges.csv')

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print("⚠️ Файлы nodes.csv или edges.csv не найдены!")
        return

    print("=== АНАЛИЗ ДАННЫХ ГРАФА ===")
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    print(f"Узлы: {len(nodes)}")
    print(f"Типы узлов: {nodes['type'].value_counts().to_dict()}")
    print()

    print(f"Рёбра: {len(edges)}")
    print(f"Уникальные источники: {edges['source'].nunique()}")
    print(f"Уникальные цели: {edges['target'].nunique()}")

    # Проверяем типы связей
    source_types = []
    target_types = []

    for _, row in edges.iterrows():
        source = str(row['source'])
        target = str(row['target'])

        if source.startswith('user_'):
            source_types.append('user')
        elif source.startswith('book_'):
            source_types.append('book')
        elif source.startswith('author_'):
            source_types.append('author')
        else:
            source_types.append('other')

        if target.startswith('user_'):
            target_types.append('user')
        elif target.startswith('book_'):
            target_types.append('book')
        elif target.startswith('author_'):
            target_types.append('author')
        else:
            target_types.append('other')

    print(f"Типы источников: {pd.Series(source_types).value_counts().to_dict()}")
    print(f"Типы целей: {pd.Series(target_types).value_counts().to_dict()}")


if __name__ == "__main__":
    check_files()
    check_graph_data()