import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import EMBEDDINGS_DIR, PLOTS_DIR, RAW_DATA_DIR
from src.visualizer import plot_embeddings


def create_all_visualizations():
    """Создаёт UMAP и t-SNE визуализации для всех экспериментов"""
    print("\n" + "=" * 50)
    print(" СОЗДАНИЕ ВИЗУАЛИЗАЦИИ ЭМБЕДДИНГОВ")
    print("=" * 50)

    nodes_path = os.path.join(RAW_DATA_DIR, 'nodes.csv')

    if not os.path.exists(nodes_path):
        print("файл nodes.csv не найден")
        return

    nodes_df = pd.read_csv(nodes_path).set_index('id')
    print(f"✓ Загружено {len(nodes_df)} узлов: {nodes_df['type'].value_counts().to_dict()}")

    experiments = [
        {'id': 'exp_001', 'title': 'Эксперимент 1 (p=1.0, q=1.0)', 'p': 1.0, 'q': 1.0},
        {'id': 'exp_002', 'title': 'Эксперимент 2 (p=2.0, q=0.5)', 'p': 2.0, 'q': 0.5},
        {'id': 'exp_003', 'title': 'Эксперимент 3 (p=0.5, q=2.0)', 'p': 0.5, 'q': 2.0}
    ]

    for exp in experiments:
        exp_id = exp['id']
        title = exp['title']

        print(f"\n→ Визуализация {exp_id}...")

        emb_path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.csv")

        if not os.path.exists(emb_path):
            print(f"Эмбеддинги не найдены: {emb_path}")
            continue

        emb_df = pd.read_csv(emb_path, index_col=0)
        print(f"  Размер эмбеддингов: {emb_df.shape}")

        common_nodes = nodes_df.index.intersection(emb_df.index)
        filtered_nodes = nodes_df.loc[common_nodes]
        filtered_emb = emb_df.loc[common_nodes]

        print(f"  Общих узлов: {len(common_nodes)}")

        if len(common_nodes) < 10:
            print(f"Слишком мало узлов для визуализации ({len(common_nodes)})")
            continue

        # UMAP
        try:
            umap_path = os.path.join(PLOTS_DIR, f"{exp_id}_umap.png")
            plot_embeddings(
                filtered_emb,
                filtered_nodes,
                method='umap',
                title=f"Проекция UMAP - {title}",
                save_path=umap_path
            )
            print(f"  ✓ UMAP сохранён: {umap_path}")
        except Exception as e:
            print(f"  Ошибка UMAP: {e}")

        # t-SNE
        try:
            tsne_path = os.path.join(PLOTS_DIR, f"{exp_id}_tsne.png")
            plot_embeddings(
                filtered_emb,
                filtered_nodes,
                method='tsne',
                title=f"Проекция t-SNE - {title}",
                save_path=tsne_path
            )
            print(f"  ✓ t-SNE сохранён: {tsne_path}")
        except Exception as e:
            print(f"  Ошибка t-SNE: {e}")

    print(f"\n✓ Визуализации созданы! Проверьте папку: {PLOTS_DIR}")
    print("=" * 50)


def create_embeddings_comparison():
    """Создает сравнительную визуализацию эмбеддингов (круговые диаграммы)"""
    print("\n" + "=" * 50)
    print(" СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
    print("=" * 50)

    experiments = ['exp_001', 'exp_002', 'exp_003']
    exp_titles = [
        'Эксперимент 1\n(p=1.0, q=1.0)',
        'Эксперимент 2\n(p=2.0, q=0.5)',
        'Эксперимент 3\n(p=0.5, q=2.0)'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Перевод типов узлов на русский
    node_type_names = {
        'user': 'Пользователь',
        'book': 'Книга',
        'author': 'Автор'
    }

    # Цвета для типов узлов
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#999999']

    for i, exp_id in enumerate(experiments):
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.csv")

        if not os.path.exists(emb_path):
            print(f"⚠ Эмбеддинги не найдены: {exp_id}")
            continue

        emb_df = pd.read_csv(emb_path, index_col=0)

        node_types = []
        for node in emb_df.index:
            node = str(node)
            if node.startswith('user_'):
                node_types.append('user')
            elif node.startswith('book_'):
                node_types.append('book')
            elif node.startswith('author_'):
                node_types.append('author')
            else:
                node_types.append('other')

        # Подсчёт типов
        type_counts = pd.Series(node_types).value_counts()

        # Перевод названий типов
        labels = [node_type_names.get(t, t) for t in type_counts.index]

        # Построение круговой диаграммы
        wedges, texts, autotexts = axes[i].pie(
            type_counts.values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors[:len(type_counts)],
            startangle=90,
            textprops={'fontsize': 10, 'weight': 'bold'}
        )

        # Форматирование процентов
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        axes[i].set_title(exp_titles[i], fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('Распределение типов узлов в эмбеддингах', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    comparison_path = os.path.join(PLOTS_DIR, 'embeddings_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Сравнение эмбеддингов сохранено: {comparison_path}")
    print("=" * 50)

    plt.show()


if __name__ == "__main__":
    create_all_visualizations()
    create_embeddings_comparison()