import os
import pandas as pd
import networkx as nx
from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, EMBEDDINGS_DIR, METRICS_DIR
from src.data_loader import load_graph
from src.simple_node2vec import SimpleNode2Vec, simple_spectral_embeddings

# Параметры экспериментов (упрощённый список для тестирования)
EXPERIMENT_PARAMETERS = [
    {'experiment_id': 'exp_001', 'p': 1.0, 'q': 1.0, 'walk_length': 10, 'num_walks': 5, 'dimensions': 64},
    {'experiment_id': 'exp_002', 'p': 2.0, 'q': 0.5, 'walk_length': 10, 'num_walks': 5, 'dimensions': 64},
    {'experiment_id': 'exp_003', 'p': 0.5, 'q': 2.0, 'walk_length': 10, 'num_walks': 5, 'dimensions': 64},
]


def run_single_experiment(graph, params, use_simple=True):
    """Запускает один эксперимент"""
    exp_id = params["experiment_id"]
    print(f"\n=== Эксперимент {exp_id} ===")
    print(f"Параметры: p={params['p']}, q={params['q']}")
    print(f"Граф: узлов={graph.number_of_nodes()}, рёбер={graph.number_of_edges()}")

    try:
        if use_simple:
            # Используем простую реализацию
            model = SimpleNode2Vec(
                graph,
                p=params["p"],
                q=params["q"],
                walk_length=params["walk_length"],
                num_walks=params["num_walks"],
                dimensions=params["dimensions"]
            ).fit()
            embeddings = model.get_embeddings()
        else:
            # Альтернативный метод
            embeddings = simple_spectral_embeddings(graph, params["dimensions"])

        if not embeddings:
            print(f" Не удалось создать эмбеддинги для {exp_id}")
            return None

        print(f"✅ Создано эмбеддингов: {len(embeddings)}")

        # Сохраняем эмбеддинги
        df_emb = pd.DataFrame.from_dict(embeddings, orient="index")
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.csv")
        df_emb.to_csv(emb_path)
        print(f" Сохранено: {emb_path}")

        return {
            "experiment_id": exp_id,
            "p": params["p"],
            "q": params["q"],
            "walk_length": params["walk_length"],
            "num_walks": params["num_walks"],
            "dimensions": params["dimensions"],
            "embeddings_file": emb_path,
            "num_embeddings": len(embeddings)
        }

    except Exception as e:
        print(f"❌ Ошибка в эксперименте {exp_id}: {e}")
        return None


def run_all_experiments():
    """Запускает все эксперименты"""
    print("🚀 Запуск экспериментов Node2Vec")

    # Загружаем граф
    try:
        graph = load_graph("graph.graphml")
        print(f"📊 Граф загружен: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер")
    except Exception as e:
        print(f"❌ Ошибка загрузки графа: {e}")
        return

    results = []

    # Запускаем эксперименты
    for i, params in enumerate(EXPERIMENT_PARAMETERS):
        print(f"\n--- Прогресс: {i + 1}/{len(EXPERIMENT_PARAMETERS)} ---")

        result = run_single_experiment(graph, params, use_simple=True)
        if result:
            results.append(result)

    # Сохраняем сводную таблицу
    if results:
        df_results = pd.DataFrame(results)
        summary_path = os.path.join(METRICS_DIR, "experiments_summary.csv")
        df_results.to_csv(summary_path, index=False)
        print(f"\n📋 Сводная таблица сохранена: {summary_path}")
        print("\n🎉 Эксперименты завершены!")

        # Показываем статистику
        print("\n📈 Статистика:")
        for _, row in df_results.iterrows():
            print(f"  {row['experiment_id']}: {row['num_embeddings']} эмбеддингов")
    else:
        print("\n❌ Ни одного успешного эксперимента")


if __name__ == "__main__":
    run_all_experiments()