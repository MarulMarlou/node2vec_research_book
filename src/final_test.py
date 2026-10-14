# final_test.py
"""
Финальный тест всех установленных библиотек
"""


def test_everything():
    """Тестируем все необходимые компоненты"""

    print("Тестирование импортов...")

    try:
        # Основные библиотеки для работы с данными
        import pandas as pd
        import numpy as np
        print("✓ Pandas и NumPy")

        # Работа с графами
        import networkx as nx
        print("✓ NetworkX")

        # Node2Vec
        try:
            from node2vec import Node2Vec
            print("✓ Node2Vec")
            node2vec_available = True
        except ImportError:
            print("⚠️ Node2Vec не найден, будем использовать альтернативную реализацию")
            node2vec_available = False

        # Gensim для Word2Vec
        try:
            import gensim
            print("✓ Gensim")
            gensim_available = True
        except ImportError:
            print("⚠️ Gensim не найден, будем использовать sklearn")
            gensim_available = False

        # Визуализация
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✓ Matplotlib и Seaborn")

        # Машинное обучение
        import sklearn
        print("✓ Scikit-learn")

        # Дополнительные
        import plotly
        from tqdm import tqdm
        print("✓ Plotly и TQDM")

        print(f"\nРезультаты:")
        print(f"Node2Vec доступен: {node2vec_available}")
        print(f"Gensim доступен: {gensim_available}")

        return node2vec_available, gensim_available

    except Exception as e:
        print(f"Ошибка: {e}")
        return False, False


def create_test_graph():
    """Создаем тестовый граф для проверки"""
    import networkx as nx

    # Создаем простой граф книг и авторов
    G = nx.Graph()

    # Добавляем узлы (книги и авторы)
    books = ["Война_и_мир", "Анна_Каренина", "Преступление_и_наказание", "Идиот"]
    authors = ["Толстой", "Достоевский"]
    genres = ["Классика", "Роман", "Драма"]

    # Добавляем узлы
    for book in books:
        G.add_node(book, type="book")
    for author in authors:
        G.add_node(author, type="author")
    for genre in genres:
        G.add_node(genre, type="genre")

    # Добавляем связи
    G.add_edge("Толстой", "Война_и_мир")
    G.add_edge("Толстой", "Анна_Каренина")
    G.add_edge("Достоевский", "Преступление_и_наказание")
    G.add_edge("Достоевский", "Идиот")

    G.add_edge("Война_и_мир", "Классика")
    G.add_edge("Война_и_мир", "Роман")
    G.add_edge("Анна_Каренина", "Классика")
    G.add_edge("Анна_Каренина", "Роман")
    G.add_edge("Преступление_и_наказание", "Классика")
    G.add_edge("Идиот", "Классика")
    G.add_edge("Идиот", "Драма")

    print(f"\n📚 Тестовый граф создан:")
    print(f"Узлов: {G.number_of_nodes()}")
    print(f"Рёбер: {G.number_of_edges()}")

    return G


if __name__ == "__main__":
    node2vec_ok, gensim_ok = test_everything()

    if not node2vec_ok or not gensim_ok:
        print("\n💡 Не все библиотеки установлены, но мы можем продолжить!")
        print("Создадим собственную реализацию Node2Vec")

    # Тестируем создание графа
    test_graph = create_test_graph()
    print("\n🎉 Готовы к работе!")
