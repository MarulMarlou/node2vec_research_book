import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import METRICS_DIR, PLOTS_DIR, EMBEDDINGS_DIR, RAW_DATA_DIR
from src.metrics import precision_recall_at_k


def load_all_results():
    """Загружает результаты всех экспериментов"""
    summary_path = os.path.join(METRICS_DIR, "experiments_summary.csv")

    if not os.path.exists(summary_path):
        print("Файл experiments_summary.csv не найден!")
        return None

    summary_df = pd.read_csv(summary_path)
    print(f"📊 Загружено {len(summary_df)} экспериментов")
    return summary_df


def calculate_all_metrics():
    """Рассчитывает метрики для всех экспериментов"""
    summary_df = load_all_results()
    if summary_df is None:
        return None

    # Загружаем рёбра для метрик
    edges_path = os.path.join(RAW_DATA_DIR, 'edges.csv')
    if not os.path.exists(edges_path):
        print("Файл edges.csv не найден!")
        return None

    edges = pd.read_csv(edges_path)

    results = []

    for _, row in summary_df.iterrows():
        exp_id = row['experiment_id']
        emb_path = row['embeddings_file']
        print(f"\nАнализируем {exp_id}...")
        if not os.path.exists(emb_path):
            print(f"Эмбеддинги не найдены: {emb_path}")
            continue

        emb_df = pd.read_csv(emb_path, index_col=0)
        metrics = precision_recall_at_k(emb_df, edges, ks=(5, 10))
        result = {
            'experiment_id': exp_id,
            'p': row['p'],
            'q': row['q'],
            'walk_length': row['walk_length'],
            'num_walks': row['num_walks'],
            'dimensions': row['dimensions'],
            'num_embeddings': row.get('num_embeddings', len(emb_df)),
            **metrics
        }
        results.append(result)
        print(f"  P@5: {metrics.get('Precision@5', 0):.4f}")
        print(f"  R@5: {metrics.get('Recall@5', 0):.4f}")
        print(f"  P@10: {metrics.get('Precision@10', 0):.4f}")
        print(f"  R@10: {metrics.get('Recall@10', 0):.4f}")
    if not results:
        print(" Нет результатов для анализа")
        return None
    results_df = pd.DataFrame(results)
    full_results_path = os.path.join(METRICS_DIR, "full_results.csv")
    results_df.to_csv(full_results_path, index=False)
    print(f"\nПолные результаты сохранены: {full_results_path}")
    return results_df


def create_comparison_plots(results_df):
    """Создаёт сравнительные графики"""
    if results_df is None or len(results_df) == 0:
        print(" Нет данных для графиков")
        return
    print("\nСоздание графиков...")
    plt.style.use('default')
    sns.set_palette("husl")
    plt.figure(figsize=(10, 6))
    for q_val in results_df['q'].unique():
        subset = results_df[results_df['q'] == q_val]
        plt.plot(subset['p'], subset['Precision@10'],
                 marker='o', label=f'q={q_val}', linewidth=2, markersize=8)

    plt.xlabel('Параметр p (return parameter)', fontsize=12)
    plt.ylabel('Precision@10', fontsize=12)
    plt.title('Зависимость Precision@10 от параметра p', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p_plot_path = os.path.join(PLOTS_DIR, 'precision_vs_p.png')
    plt.savefig(p_plot_path, dpi=300, bbox_inches='tight')
    print(f"График p: {p_plot_path}")
    plt.show()

    plt.figure(figsize=(10, 6))
    for p_val in results_df['p'].unique():
        subset = results_df[results_df['p'] == p_val]
        plt.plot(subset['q'], subset['Recall@10'],
                 marker='s', label=f'p={p_val}', linewidth=2, markersize=8)

    plt.xlabel('Параметр q (in-out parameter)', fontsize=12)
    plt.ylabel('Recall@10', fontsize=12)
    plt.title('Зависимость Recall@10 от параметра q', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    q_plot_path = os.path.join(PLOTS_DIR, 'recall_vs_q.png')
    plt.savefig(q_plot_path, dpi=300, bbox_inches='tight')
    print(f"   График q: {q_plot_path}")
    plt.show()

    # 3. Тепловая карта всех метрик
    plt.figure(figsize=(12, 8))
    pivot_data = results_df.pivot_table(
        values=['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10'],
        index='p',
        columns='q',
        aggfunc='mean'
    )
    if len(results_df['p'].unique()) > 1 and len(results_df['q'].unique()) > 1:
        for i, metric in enumerate(['Precision@5', 'Precision@10', 'Recall@5', 'Recall@10']):
            plt.subplot(2, 2, i + 1)
            sns.heatmap(pivot_data[metric], annot=True, fmt='.3f', cmap='viridis')
            plt.title(f'{metric}')

    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, 'metrics_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"   Тепловая карта: {heatmap_path}")
    plt.show()


def generate_summary_table(results_df):
    """Создаёт итоговую сводную таблицу"""
    if results_df is None:
        return

    print("\nСоздание сводной таблицы...")
    best_precision5 = results_df.loc[results_df['Precision@5'].idxmax()]
    best_precision10 = results_df.loc[results_df['Precision@10'].idxmax()]
    best_recall5 = results_df.loc[results_df['Recall@5'].idxmax()]
    best_recall10 = results_df.loc[results_df['Recall@10'].idxmax()]

    summary_text = f"""
# РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ NODE2VEC

## Общая статистика
- Всего экспериментов: {len(results_df)}
- Диапазон параметра p: {results_df['p'].min():.1f} - {results_df['p'].max():.1f}
- Диапазон параметра q: {results_df['q'].min():.1f} - {results_df['q'].max():.1f}

## Лучшие результаты

### Precision@5
**Эксперимент:** {best_precision5['experiment_id']}
**Параметры:** p={best_precision5['p']:.1f}, q={best_precision5['q']:.1f}
**Результат:** {best_precision5['Precision@5']:.4f}

### Precision@10  
**Эксперимент:** {best_precision10['experiment_id']}
**Параметры:** p={best_precision10['p']:.1f}, q={best_precision10['q']:.1f}
**Результат:** {best_precision10['Precision@10']:.4f}

### Recall@5
**Эксперимент:** {best_recall5['experiment_id']}
**Параметры:** p={best_recall5['p']:.1f}, q={best_recall5['q']:.1f}
**Результат:** {best_recall5['Recall@5']:.4f}

### Recall@10
**Эксперимент:** {best_recall10['experiment_id']}
**Параметры:** p={best_recall10['p']:.1f}, q={best_recall10['q']:.1f}
**Результат:** {best_recall10['Recall@10']:.4f}

## Средние значения метрик
- Precision@5: {results_df['Precision@5'].mean():.4f} ± {results_df['Precision@5'].std():.4f}
- Precision@10: {results_df['Precision@10'].mean():.4f} ± {results_df['Precision@10'].std():.4f}
- Recall@5: {results_df['Recall@5'].mean():.4f} ± {results_df['Recall@5'].std():.4f}  
- Recall@10: {results_df['Recall@10'].mean():.4f} ± {results_df['Recall@10'].std():.4f}

## Выводы

1. **Влияние параметра p:** {analyze_parameter_effect(results_df, 'p')}

2. **Влияние параметра q:** {analyze_parameter_effect(results_df, 'q')}

3. **Рекомендуемые параметры:** 
   - Для максимального Precision@10: p={best_precision10['p']:.1f}, q={best_precision10['q']:.1f}
   - Для максимального Recall@10: p={best_recall10['p']:.1f}, q={best_recall10['q']:.1f}
"""
    report_path = os.path.join(METRICS_DIR, "analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f" Отчёт сохранён: {report_path}")
    print(summary_text)


def analyze_parameter_effect(df, param):
    """Анализирует влияние параметра на метрики"""
    if df[param].nunique() < 2:
        return f"Недостаточно вариаций параметра {param} для анализа"

    corr_p10 = df[param].corr(df['Precision@10'])
    corr_r10 = df[param].corr(df['Recall@10'])

    if abs(corr_p10) > 0.5:
        p_effect = f"сильное {'положительное' if corr_p10 > 0 else 'отрицательное'} влияние на Precision@10"
    elif abs(corr_p10) > 0.3:
        p_effect = f"умеренное {'положительное' if corr_p10 > 0 else 'отрицательное'} влияние на Precision@10"
    else:
        p_effect = "слабое влияние на Precision@10"

    if abs(corr_r10) > 0.5:
        r_effect = f"сильное {'положительное' if corr_r10 > 0 else 'отрицательное'} влияние на Recall@10"
    elif abs(corr_r10) > 0.3:
        r_effect = f"умеренное {'положительное' if corr_r10 > 0 else 'отрицательное'} влияние на Recall@10"
    else:
        r_effect = "слабое влияние на Recall@10"

    return f"{p_effect}, {r_effect}"


def main():
    """Главная функция анализа"""
    print(" АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ NODE2VEC")
    print("=" * 50)
    results_df = calculate_all_metrics()

    if results_df is not None:
        create_comparison_plots(results_df)
        generate_summary_table(results_df)
        print("\n Анализ завершён! Проверьте папки:")
        print(f"   Графики: {PLOTS_DIR}")
        print(f"   Метрики: {METRICS_DIR}")
    else:
        print(" Анализ не удался - нет данных")


if __name__ == "__main__":
    main()