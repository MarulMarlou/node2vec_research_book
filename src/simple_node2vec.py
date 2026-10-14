import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from collections import defaultdict
import random


class SimpleNode2Vec:
    """Упрощённая реализация Node2Vec для курсовой работы"""

    def __init__(self, graph, p=1, q=1, walk_length=80, num_walks=10, dimensions=128):
        self.graph = graph
        self.p = p  # return parameter
        self.q = q  # in-out parameter  
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.dimensions = dimensions
        self.walks = []
        self.embeddings = {}

    def generate_walks(self):
        """Генерирует случайные блуждания по графу с учётом параметров p и q"""
        print(f"Генерация {self.num_walks} блужданий длиной {self.walk_length}...")

        nodes = list(self.graph.nodes())
        for walk_iter in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.node2vec_walk(node)
                if len(walk) > 1:
                    self.walks.append(walk)

        print(f"Сгенерировано {len(self.walks)} блужданий")

    def node2vec_walk(self, start_node):
        """Одно блуждание Node2Vec от узла start_node"""
        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))

            if len(neighbors) == 0:
                break

            if len(walk) == 1:
                # Первый шаг - равномерно случайный
                next_node = random.choice(neighbors)
            else:
                # Используем параметры p и q
                prev = walk[-2]
                probs = []

                for neighbor in neighbors:
                    if neighbor == prev:
                        # Возврат назад
                        probs.append(1 / self.p)
                    elif self.graph.has_edge(neighbor, prev):
                        # Узел на расстоянии 1 от предыдущего
                        probs.append(1)
                    else:
                        # Узел на расстоянии 2 от предыдущего  
                        probs.append(1 / self.q)

                # Нормализация вероятностей
                total = sum(probs)
                probs = [p / total for p in probs]

                # Выбираем следующий узел
                next_node = np.random.choice(neighbors, p=probs)

            walk.append(next_node)

        return [str(node) for node in walk]

    def learn_embeddings(self):
        """Обучает эмбеддинги на основе блужданий (упрощённая версия Word2Vec)"""
        print("Обучение эмбеддингов...")

        # Создаём словарь узлов
        vocab = set()
        for walk in self.walks:
            vocab.update(walk)
        vocab = list(vocab)

        if len(vocab) == 0:
            print("Нет узлов для обучения!")
            return

        print(f"Словарь: {len(vocab)} узлов")

        # Инициализация случайных эмбеддингов
        np.random.seed(42)
        embeddings = {}
        for node in vocab:
            embeddings[node] = np.random.randn(self.dimensions) * 0.1

        # Простое обучение Skip-gram (очень упрощённое)
        window_size = 5
        learning_rate = 0.01
        epochs = 5

        for epoch in range(epochs):
            print(f"Эпоха {epoch + 1}/{epochs}")
            for walk in self.walks:
                for i, center in enumerate(walk):
                    # Контекстные слова
                    start = max(0, i - window_size)
                    end = min(len(walk), i + window_size + 1)

                    context = walk[start:i] + walk[i + 1:end]

                    for context_word in context:
                        # Упрощённое обновление (градиентный спуск)
                        error = embeddings[center] - embeddings[context_word]
                        embeddings[center] -= learning_rate * error * 0.01
                        embeddings[context_word] += learning_rate * error * 0.01

        self.embeddings = embeddings
        print(f"Обучено эмбеддингов: {len(self.embeddings)}")

    def fit(self):
        """Полный цикл обучения"""
        self.generate_walks()
        self.learn_embeddings()
        return self

    def get_embeddings(self):
        """Возвращает словарь эмбеддингов node_id -> vector"""
        return self.embeddings


# Альтернативная реализация через NetworkX + sklearn
def simple_spectral_embeddings(graph, dimensions=128):
    """Простые спектральные эмбеддинги как альтернатива Node2Vec"""
    print(f"Генерация спектральных эмбеддингов размерности {dimensions}...")

    # Матрица смежности
    adj_matrix = nx.adjacency_matrix(graph)

    # Собственные векторы лапласиана
    try:
        from sklearn.manifold import SpectralEmbedding
        embedding = SpectralEmbedding(n_components=min(dimensions, adj_matrix.shape[0] - 1))
        node_embeddings = embedding.fit_transform(adj_matrix)
    except:
        # Fallback к PCA
        pca = PCA(n_components=min(dimensions, adj_matrix.shape[0] - 1))
        node_embeddings = pca.fit_transform(adj_matrix.toarray())

    # Дополняем до нужной размерности если необходимо
    if node_embeddings.shape[1] < dimensions:
        padding = np.random.randn(node_embeddings.shape[0],
                                  dimensions - node_embeddings.shape[1]) * 0.01
        node_embeddings = np.hstack([node_embeddings, padding])

    # Создаём словарь
    nodes = list(graph.nodes())
    embeddings = {str(nodes[i]): node_embeddings[i] for i in range(len(nodes))}

    print(f"Создано эмбеддингов: {len(embeddings)}")
    return embeddings