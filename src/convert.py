import pandas as pd
import os
from src.config import RAW_DATA_DIR


def convert_bx_to_graph():
    print("Конвертация Book-Crossing данных...")

    try:
        print("Загрузка BX-Books.csv...")
        try:
            books = pd.read_csv(os.path.join(RAW_DATA_DIR, "BX-Books.csv"), sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
        except:
            books = pd.read_csv(os.path.join(RAW_DATA_DIR, "BX-Books.csv"), sep=';', encoding='utf-8', on_bad_lines='skip', low_memory=False)

        print("Загрузка BX-Book-Ratings.csv...")
        try:
            ratings = pd.read_csv(os.path.join(RAW_DATA_DIR, "BX-Book-Ratings.csv"), sep=';', encoding='latin-1')
        except:
            ratings = pd.read_csv(os.path.join(RAW_DATA_DIR, "BX-Book-Ratings.csv"), sep=';', encoding='utf-8')

        print(f"Загружено {len(books)} книг и {len(ratings)} рейтингов")

        popular_books = ratings[ratings['Book-Rating'] > 0]['ISBN'].value_counts()
        top_books = popular_books.head(5000).index
        active_users = ratings[ratings['Book-Rating'] > 0]['User-ID'].value_counts()
        top_users = active_users.head(1000).index
        filtered_books = books[books['ISBN'].isin(top_books)].copy()
        filtered_ratings = ratings[
            (ratings['ISBN'].isin(top_books)) &
            (ratings['User-ID'].isin(top_users)) &
            (ratings['Book-Rating'] > 0)
            ].copy()
        print(f"После фильтрации: {len(filtered_books)} книг, {len(filtered_ratings)} рейтингов")

        nodes = []
        for _, row in filtered_books.iterrows():
            if pd.notna(row['ISBN']) and pd.notna(row['Book-Title']):
                nodes.append({
                    'id': f"book_{row['ISBN']}",
                    'type': 'book',
                    'label': str(row['Book-Title'])[:100]  # Обрезаем длинные названия
                })
        unique_authors = filtered_books['Book-Author'].dropna().unique()
        for author in unique_authors:
            nodes.append({
                'id': f"author_{author}",
                'type': 'author',
                'label': str(author)[:50]
            })
        for user_id in top_users:
            nodes.append({
                'id': f"user_{user_id}",
                'type': 'user',
                'label': f"User {user_id}"
            })
        nodes_df = pd.DataFrame(nodes).drop_duplicates(subset='id')
        nodes_df = nodes_df.dropna()
        print(f"Создано {len(nodes_df)} узлов")

        edges = []
        for _, row in filtered_books.iterrows():
            if pd.notna(row['ISBN']) and pd.notna(row['Book-Author']):
                edges.append({
                    'source': f"book_{row['ISBN']}",
                    'target': f"author_{row['Book-Author']}",
                    'weight': 1
                })
        for _, row in filtered_ratings.iterrows():
            if pd.notna(row['ISBN']) and pd.notna(row['User-ID']):
                normalized_rating = max(0.1, float(row['Book-Rating']) / 10.0)
                edges.append({
                    'source': f"user_{row['User-ID']}",
                    'target': f"book_{row['ISBN']}",
                    'weight': normalized_rating
                })
        edges_df = pd.DataFrame(edges)
        edges_df = edges_df.dropna()

        print(f"Создано {len(edges_df)} рёбер")
        nodes_path = os.path.join(RAW_DATA_DIR, 'nodes.csv')
        edges_path = os.path.join(RAW_DATA_DIR, 'edges.csv')

        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)

        print(f"Файлы сохранены:")
        print(f"  nodes.csv: {nodes_path}")
        print(f"  edges.csv: {edges_path}")
        print(f"Статистика графа:")
        print(f"  Узлов: {len(nodes_df)}")
        print(f"  Рёбер: {len(edges_df)}")
        print(f"  Типы узлов: {nodes_df['type'].value_counts().to_dict()}")

        return True

    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
        print("Убедитесь, что файлы BX-Books.csv и BX-Book-Ratings.csv находятся в папке data/raw/")
        return False

    except Exception as e:
        print(f"Ошибка при конвертации: {e}")
        return False


if __name__ == "__main__":
    success = convert_bx_to_graph()
    if success:
        print("\nКонвертация завершена успешно!")
    else:
        print("\nОшибка при конвертации")