
# app.py
"""
Streamlit веб-приложение рекомендательной системы на графах знаний.

Запуск:
    streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ── пути ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "results", "embeddings")
METRICS_CSV   = os.path.join(BASE_DIR, "results", "metrics", "all_experiments.csv")
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")

st.set_page_config(
    page_title="Book Recommender · KG",
    page_icon="📚",
    layout="wide",
)


#  Кэшированные загрузчики

@st.cache_data(show_spinner="Загрузка данных о книгах …")
def load_books() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "nodes.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["id", "type", "label", "isbn"])
    df = pd.read_csv(path)
    return df[df["type"] == "book"].reset_index(drop=True)


@st.cache_data(show_spinner="Загрузка результатов экспериментов …")
def load_results() -> pd.DataFrame:
    if not os.path.exists(METRICS_CSV):
        return pd.DataFrame()
    return pd.read_csv(METRICS_CSV)


@st.cache_data(show_spinner="Загрузка эмбеддингов …")
def load_embeddings(exp_id: str, algo: str) -> tuple:
    """Возвращает (emb_df, rated_vec_or_None)"""
    npy   = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    idx_f = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_idx.txt")
    if not os.path.exists(npy) or not os.path.exists(idx_f):
        return None, None
    vals = np.load(npy)
    with open(idx_f, encoding="utf-8") as f:
        index = [l.strip() for l in f]
    emb_df = pd.DataFrame(vals, index=index)

    rated_vec = None
    if algo == "TransE":
        rvec_path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_rated_vec.npy")
        if os.path.exists(rvec_path):
            rated_vec = np.load(rvec_path)
    return emb_df, rated_vec


#  Рекомендации

def recommend_n2v(book_id: str, emb_df: pd.DataFrame, books_df: pd.DataFrame,
                  top_k: int = 10) -> pd.DataFrame:
    """Косинусное сходство книги ↔ книги (Node2Vec)."""
    book_ids = [n for n in emb_df.index if str(n).startswith("book_")]
    if book_id not in emb_df.index or not book_ids:
        return pd.DataFrame()
    query_vec = emb_df.loc[book_id].values.reshape(1, -1)
    book_emb  = emb_df.loc[book_ids].values
    sims      = cosine_similarity(query_vec, book_emb)[0]
    top_idx   = np.argsort(-sims)[:top_k + 1]
    recs = []
    for i in top_idx:
        bid = book_ids[i]
        if bid == book_id:
            continue
        row = books_df[books_df["id"] == bid]
        title  = row["label"].values[0] if len(row) else bid
        recs.append({"book_id": bid, "title": title,
                     "similarity": round(float(sims[i]), 4)})
        if len(recs) == top_k:
            break
    return pd.DataFrame(recs)


def recommend_transe(book_id: str, emb_df: pd.DataFrame,
                     rated_vec: np.ndarray, books_df: pd.DataFrame,
                     top_k: int = 10) -> pd.DataFrame:
    """TransE: score = -|| t_book_query + r_rated - t_b ||_2 (book↔book proxy)."""
    book_ids = [n for n in emb_df.index if str(n).startswith("book_")]
    if book_id not in emb_df.index or not book_ids:
        return pd.DataFrame()
    query_vec = emb_df.loc[book_id].values
    recs = []
    for bid in book_ids:
        if bid == book_id:
            continue
        t_b   = emb_df.loc[bid].values
        score = -np.linalg.norm(query_vec + (rated_vec if rated_vec is not None else 0) - t_b)
        recs.append((bid, float(score)))
    recs.sort(key=lambda x: x[1], reverse=True)
    rows = []
    for bid, sc in recs[:top_k]:
        row   = books_df[books_df["id"] == bid]
        title = row["label"].values[0] if len(row) else bid
        rows.append({"book_id": bid, "title": title, "score": round(sc, 4)})
    return pd.DataFrame(rows)


#  UMAP / t-SNE для Plotly

@st.cache_data(show_spinner="Вычисляем 2D-проекцию …")
def compute_projection(exp_id: str, algo: str, method: str = "umap"):
    # загружаем эмбеддинги
    try:
        if algo == "Node2Vec":
            import node2vec_runner as n2v
            emb_df = n2v.load_embeddings(exp_id)
        else:
            import transe_runner as transe
            emb_raw, rated_vec = transe.load_embeddings(exp_id)
            emb_df = transe.build_transe_emb_df_for_metrics(emb_raw, rated_vec)
    except Exception as e:
        st.error(f"Ошибка загрузки эмбеддингов {exp_id}: {e}")
        return None

    # подвыборка до 2000 точек
    sample = emb_df.sample(min(500, len(emb_df)), random_state=42)

    # определяем тип каждого узла по префиксу node_id
    def get_type(node_id: str) -> str:
        s = str(node_id)
        if s.startswith("user"):   return "Пользователь"
        if s.startswith("book"):   return "Книга"
        if s.startswith("author"): return "Автор"
        return "Другое"

    node_types = [get_type(n) for n in sample.index]

    # 2D проекция
    try:
        from sklearn.manifold import TSNE
        proj = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,
            n_iter=500,  # поменьше итераций
            learning_rate="auto",
            init="pca",
        ).fit_transform(sample.values)
    except Exception as e:
        st.error(f"Ошибка при вычислении t-SNE: {e}")
        return None

    return pd.DataFrame({
        "x":    proj[:, 0],
        "y":    proj[:, 1],
        "type": node_types,
        "id":   sample.index.tolist(),
    })


#  UI — Sidebar

st.sidebar.title("⚙️ Настройки")

results_df = load_results()

algo = st.sidebar.selectbox("Алгоритм", ["Node2Vec", "TransE"])

if results_df.empty:
    st.sidebar.warning("Файл all_experiments.csv не найден. Запустите run_experiments.py.")
    best_configs = []
else:
    algo_df = results_df[results_df["algorithm"] == algo].copy()
    if algo_df.empty:
        best_configs = []
    else:
        p10_col = "p@10" if "p@10" in algo_df.columns else algo_df.columns[3]
        algo_df_sorted = algo_df.sort_values(p10_col, ascending=False)
        best_configs = algo_df_sorted["experiment_id"].tolist()

selected_exp = st.sidebar.selectbox(
    "Конфигурация (лучшие по P@10)",
    best_configs if best_configs else ["—"]
)
top_k = st.sidebar.slider("Top-K рекомендаций", 5, 20, 10)

exp_id = selected_exp if selected_exp != "—" else None

tab_rec, tab_viz, tab_res = st.tabs([
    "📚 Рекомендации",
    "🗺 Визуализация",
    "📊 Результаты экспериментов",
])


# ── Tab 1: Рекомендации ───────────────────────────────────────────────────────
with tab_rec:
    st.header("📚 Поиск и рекомендации книг")

    books_df = load_books()
    query    = st.text_input("Введите название книги (поиск по подстроке):", "")

    if query:
        matches = books_df[books_df["label"].str.contains(query, case=False, na=False)]
        if matches.empty:
            st.info("Книги не найдены. Попробуйте другой запрос.")
        else:
            selected_book = st.selectbox(
                f"Найдено {len(matches)} книг. Выберите:",
                matches["label"].tolist()
            )
            book_row = matches[matches["label"] == selected_book].iloc[0]
            book_id  = book_row["id"]

            st.markdown(f"**Выбрана книга:** `{selected_book}` (id: `{book_id}`)")

            if selected_exp and selected_exp != "—":
                with st.spinner("Генерируем рекомендации …"):
                    emb_df, rated_vec = load_embeddings(selected_exp, algo)
                    if emb_df is None:
                        st.error(f"Эмбеддинги для {selected_exp} не найдены. "
                                 "Запустите run_experiments.py.")
                    else:
                        if algo == "Node2Vec":
                            recs = recommend_n2v(book_id, emb_df, books_df, top_k)
                            score_col = "similarity"
                        else:
                            recs = recommend_transe(book_id, emb_df, rated_vec,
                                                    books_df, top_k)
                            score_col = "score"

                        if recs.empty:
                            st.warning("Не удалось построить рекомендации.")
                        else:
                            st.success(f"Топ-{top_k} рекомендаций ({algo} / {selected_exp})")
                            disp = recs[["title", score_col]].copy()
                            disp.index = range(1, len(disp) + 1)
                            disp.columns = ["Название книги", "Оценка сходства"]
                            st.dataframe(disp, use_container_width=True)
            else:
                st.info("Выберите конфигурацию в боковой панели.")


# ── Tab 2: Визуализация ───────────────────────────────────────────────────────
with tab_viz:
    st.header("🗺 Визуализация эмбеддингов")

    if not exp_id:
        st.info("Выберите конфигурацию в боковой панели.")
    else:
        method = st.radio(
            "Метод",
            ["tsne"],  # временно только t-SNE
            format_func=lambda x: "t-SNE",
            horizontal=True,
        )

        proj_df = compute_projection(exp_id, algo, method)

    if proj_df is not None:
        import plotly.express as px
        color_map = {
            "Пользователь": "#FF6B6B",
            "Книга":        "#4ECDC4",
            "Автор":        "#45B7D1",
            "Другое":       "#999999",
        }
        fig = px.scatter(
            proj_df, x="x", y="y",
            color="type",
            color_discrete_map=color_map,
            hover_data={"id": True, "type": True, "x": False, "y": False},
            labels={"x": "Компонента 1", "y": "Компонента 2", "type": "Тип"},
            title=f"{method.upper()} — {algo} ({exp_id})",
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Не удалось построить проекцию.")


# ── Tab 3: Результаты экспериментов ──────────────────────────────────────────
with tab_res:
    st.header("📊 Все эксперименты")

    if results_df.empty:
        st.warning("Нет данных. Запустите run_experiments.py.")
    else:
        algo_filter = st.multiselect(
            "Фильтр по алгоритму",
            options=results_df["algorithm"].unique().tolist(),
            default=results_df["algorithm"].unique().tolist(),
        )
        filtered = results_df[results_df["algorithm"].isin(algo_filter)]

        metric_cols = [c for c in results_df.columns
                       if any(c.startswith(p) for p in ("p@","r@","f1@","ndcg@","mrr","auc"))]
        sort_metric = st.selectbox("Сортировать по", metric_cols,
                                   index=metric_cols.index("p@10") if "p@10" in metric_cols else 0)

        filtered_sorted = filtered.sort_values(sort_metric, ascending=False)
        st.dataframe(filtered_sorted, use_container_width=True)

        # Барплот
        if not filtered_sorted.empty and sort_metric in filtered_sorted.columns:
            fig2 = px.bar(
                filtered_sorted,
                x="experiment_id", y=sort_metric,
                color="algorithm",
                color_discrete_map={"Node2Vec": "#4ECDC4", "TransE": "#FF6B6B"},
                title=f"{sort_metric} по всем экспериментам",
                height=400,
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
