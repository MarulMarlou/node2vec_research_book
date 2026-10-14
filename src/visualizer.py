"""
t-SNE и UMAP визуализации эмбеддингов.
Все функции сохраняют PNG в results/plots/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

try:
    import umap as umap_lib
    _UMAP = True
except ImportError:
    _UMAP = False
    print("[visualize] umap-learn не установлен: pip install umap-learn")

from config import PLOTS_DIR


# ─────────────────────────── палитра ──────────────────────────────────────────
_COLORS = {"user": "#FF6B6B", "book": "#4ECDC4", "author": "#45B7D1"}
_LABELS = {"user": "Пользователь", "book": "Книга", "author": "Автор"}


def _node_type(nid: str) -> str:
    nid = str(nid)
    if nid.startswith("user_"):   return "user"
    if nid.startswith("book_"):   return "book"
    if nid.startswith("author_"): return "author"
    return "other"


def _subsample(emb_df: pd.DataFrame, max_pts: int = 3000) -> pd.DataFrame:
    """Случайная подвыборка для ускорения t-SNE/UMAP."""
    if len(emb_df) <= max_pts:
        return emb_df
    return emb_df.sample(max_pts, random_state=42)


# ─────────────────────────── core plot ────────────────────────────────────────

def _scatter(proj: np.ndarray, index: list, title: str, save_path: str,
             highlight: list = None):
    types  = [_node_type(n) for n in index]
    fig, ax = plt.subplots(figsize=(10, 8))

    for t in ("user", "book", "author"):
        mask = [i for i, tt in enumerate(types) if tt == t]
        if not mask:
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=_COLORS[t], label=_LABELS[t],
                   s=12, alpha=0.5, linewidths=0)

    if highlight:
        hi_mask = [i for i, n in enumerate(index) if n in set(highlight)]
        if hi_mask:
            ax.scatter(proj[hi_mask, 0], proj[hi_mask, 1],
                       c="gold", edgecolors="black", s=80, zorder=5,
                       label="Рекомендовано")

    patches = [mpatches.Patch(color=_COLORS[t], label=_LABELS[t])
               for t in ("user", "book", "author")]
    ax.legend(handles=patches, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Компонента 1"); ax.set_ylabel("Компонента 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] сохранено → {save_path}")


# ─────────────────────────── public API ───────────────────────────────────────

def plot_tsne(emb_df: pd.DataFrame, title: str, save_name: str,
              highlight: list = None):
    sub  = _subsample(emb_df)
    proj = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(sub) - 1)).fit_transform(sub.values)
    path = os.path.join(PLOTS_DIR, save_name)
    _scatter(proj, list(sub.index), title, path, highlight)
    return path


def plot_umap(emb_df: pd.DataFrame, title: str, save_name: str,
              highlight: list = None):
    if not _UMAP:
        print("[visualize] UMAP пропущен (не установлен).")
        return None
    sub     = _subsample(emb_df)
    reducer = umap_lib.UMAP(n_components=2, random_state=42,
                            n_neighbors=15, min_dist=0.1)
    proj    = reducer.fit_transform(sub.values)
    path    = os.path.join(PLOTS_DIR, save_name)
    _scatter(proj, list(sub.index), title, path, highlight)
    return path


# ─────────────────────────── batch charts ─────────────────────────────────────

def plot_barplot(results_df: pd.DataFrame, metric: str, save_name: str):
    """Барплот метрики по всем экспериментам (Node2Vec + TransE)."""
    fig, ax = plt.subplots(figsize=(max(12, len(results_df) * 0.5), 6))
    colors  = ["#4ECDC4" if a == "Node2Vec" else "#FF6B6B"
               for a in results_df["algorithm"]]
    bars = ax.bar(results_df["experiment_id"], results_df[metric],
                  color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("Эксперимент"); ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} по всем экспериментам", fontweight="bold")
    ax.set_xticklabels(results_df["experiment_id"], rotation=45, ha="right", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4ECDC4", label="Node2Vec"),
                        Patch(color="#FF6B6B", label="TransE")])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {save_name}")
    return path


def plot_trend_dimensions(results_df: pd.DataFrame, save_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_cols = ["p@10", "r@10", "ndcg@10"]
    metric_names = ["Precision@10", "Recall@10", "NDCG@10"]

    for ax, col, name in zip(axes, metrics_cols, metric_names):
        for algo, color in [("Node2Vec", "#4ECDC4"), ("TransE", "#FF6B6B")]:
            sub = results_df[results_df["algorithm"] == algo].copy()
            if sub.empty or col not in sub.columns or "dimension_size" not in sub.columns:
                continue
            grouped = sub.groupby("dimension_size")[col].mean().reset_index()
            grouped = grouped.sort_values("dimension_size")
            ax.plot(grouped["dimension_size"], grouped[col],
                    marker="o", label=algo, color=color, linewidth=2)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Размерность")
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Зависимость метрик от размерности эмбеддингов",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {save_name}")
    return path


def generate_all_plots(results_df: pd.DataFrame,
                       best_n2v_emb: pd.DataFrame = None,
                       best_transe_emb: pd.DataFrame = None):
    print("\n[visualize] Генерация графиков …")

    plot_barplot(results_df, "p@10",    "barplot_precision10.png")
    plot_barplot(results_df, "r@10",    "barplot_recall10.png")
    plot_barplot(results_df, "ndcg@10", "barplot_ndcg10.png")
    plot_trend_dimensions(results_df,   "trend_dimensions.png")

    if best_n2v_emb is not None:
        plot_tsne(best_n2v_emb, "t-SNE — лучший Node2Vec", "tsne_best_node2vec.png")
        plot_umap(best_n2v_emb, "UMAP — лучший Node2Vec",  "umap_best_node2vec.png")

    if best_transe_emb is not None:
        plot_tsne(best_transe_emb, "t-SNE — лучший TransE", "tsne_best_transe.png")
        plot_umap(best_transe_emb, "UMAP — лучший TransE",  "umap_best_transe.png")

    print("[visualize] Все графики сохранены в", PLOTS_DIR)
