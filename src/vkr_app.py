"""Streamlit front-end for the VKR explainable book recommendation prototype.

From repository root: streamlit run src/vkr_app.py
Generate artifacts first with python -m src.vkr_pipeline.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.vkr_dashboard import (ArtifactsError, compare_metadata_baseline,
                               list_books, load_dashboard_assets,
                               path_chart_data, recommend_for_display)


@st.cache_resource(show_spinner="Читаем подготовленный граф знаний и эмбеддинги…")
def cached_assets(processed_dir: str, results_dir: str):
    return load_dashboard_assets(Path(processed_dir), Path(results_dir))


def draw_evidence_path(chart_data: dict) -> go.Figure:
    """Plot only verified edge facts. Directions are given in the evidence table."""
    nodes = chart_data["nodes"]
    edges = chart_data["edges"]
    by_id = {n["id"]: n for n in nodes}
    colors = {"book": "#3B82F6", "author": "#10B981", "publisher": "#F59E0B",
              "genre": "#8B5CF6", "user": "#64748B"}
    fig = go.Figure()
    for evidence in edges:
        a, b = by_id[evidence["from_id"]], by_id[evidence["to_id"]]
        fig.add_trace(go.Scatter(x=[a["x"], b["x"]], y=[a["y"], b["y"]],
                                 mode="lines", line=dict(width=2, color="#94A3B8"),
                                 hoverinfo="skip", showlegend=False))
        x_mid, y_mid = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
        fig.add_annotation(x=x_mid, y=y_mid, text=evidence["relation"],
                           showarrow=False, yshift=13,
                           font=dict(size=10, color="#64748B"))
    for entity_type in sorted({n["type"] for n in nodes}):
        group = [n for n in nodes if n["type"] == entity_type]
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in group], y=[n["y"] for n in group], mode="markers+text",
            text=[n["label"][:26] + ("…" if len(n["label"]) > 26 else "") for n in group],
            textposition="bottom center", name=entity_type,
            marker=dict(size=22, color=colors.get(entity_type, "#64748B"),
                        line=dict(width=1, color="white")),
            customdata=[[n["id"], n["label"]] for n in group],
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]}<extra></extra>"))
    fig.update_layout(height=370, margin=dict(l=15, r=15, t=25, b=55),
                      xaxis=dict(visible=False, range=[-0.35, len(nodes)-0.65]),
                      yaxis=dict(visible=False, range=[-0.5, 1.0]),
                      legend=dict(orientation="h", y=1.1),
                      hovermode="closest")
    return fig


def render() -> None:
    st.set_page_config(page_title="Объяснимые рекомендации книг · ВКР", page_icon="📚",
                       layout="wide")
    st.title("📚 Объяснимая рекомендательная система книг")
    st.caption("Исследовательский прототип ВКР · Node2Vec и объясняющие пути в графе знаний")

    with st.sidebar:
        st.header("Данные и модель")
        processed_dir = st.text_input("Каталог обработанных данных", value=str(ROOT / "data" / "processed"))
        results_dir = st.text_input("Каталог результатов", value=str(ROOT / "results"))
        if st.button("Обновить загруженные артефакты"):
            st.cache_resource.clear()
            st.rerun()

    try:
        assets = cached_assets(processed_dir, results_dir)
    except ArtifactsError as exc:
        st.error(f"Модель пока не подготовлена: {exc}")
        st.code("python -m src.vkr_pipeline --raw-dir data/raw --processed-dir data/processed --results-dir results")
        st.info("Для запуска потребуются BX-Books.csv и BX-Book-Ratings.csv в data/raw/. "
                "Старые эмбеддинги курсовой не используются.")
        st.stop()

    st.sidebar.success("Граф и модель загружены")
    st.sidebar.caption(f"Узлов: {assets.graph.number_of_nodes():,} · "
                       f"Связей: {assets.graph.number_of_edges():,} · "
                       f"Эмбеддингов: {len(assets.embeddings):,}")

    st.header("Рекомендации по выбранной книге")
    query = st.text_input("Поиск по названию или ISBN", placeholder="Введите часть названия книги…")
    choices = list_books(assets.graph, query=query, limit=100)
    if not choices:
        st.warning("Нет книг, соответствующих поиску. Попробуйте другую строку.")
        st.stop()
    lookup = {b["id"]: b for b in choices}
    source = st.selectbox("Исходная книга", [b["id"] for b in choices],
                          format_func=lambda bid: f'{lookup[bid]["title"]} · ISBN {lookup[bid]["isbn"]}')

    with st.sidebar:
        st.header("Алгоритм объяснения")
        algorithm = st.selectbox("Поиск пути", ["shortest", "informative"],
                                 format_func=lambda m: {"shortest": "Кратчайший допустимый путь",
                                                        "informative": "Наиболее информативный путь"}[m])
        mode = st.radio("Режим", ["posthoc", "rerank"],
                        format_func=lambda m: {"posthoc": "Объяснения без изменения выдачи",
                                               "rerank": "Переранжирование с объяснением"}[m])
        alpha = st.slider("Вес объяснения (α)", 0.0, 1.0, 0.2, 0.05,
                          disabled=mode != "rerank")
        max_depth = st.select_slider("Максимальная длина пути (рёбер)", options=[2, 4], value=4)
        top_k = st.slider("Число рекомендаций", min_value=3, max_value=20, value=10)
        pool_size = st.slider("Кандидатов для переранжирования", min_value=top_k,
                              max_value=50, value=max(top_k, 25))

    with st.spinner("Формируем рекомендации и проверяем пути в графе…"):
        response = recommend_for_display(assets, source, method=algorithm, mode=mode,
                                         alpha=alpha, top_k=top_k,
                                         candidate_pool=pool_size, max_depth=max_depth)

    results = response["results"]
    if not results:
        st.warning("Для выбранной книги не найдены кандидаты с доступными эмбеддингами.")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Рекомендаций", len(results))
    b.metric("С объяснением", f'{response["explanation_coverage"]:.0%}')
    c.metric("Среднее время объяснения", f'{response["average_explanation_time_s"]*1000:.1f} мс')
    if response["truncated_count"]:
        st.warning(f'Для {response["truncated_count"]} рекомендаций поиск ограничен лимитом обхода: '
                   "не гарантирован наиболее информативный путь во всём графе.")

    tab_main, tab_graph, tab_baseline, tab_eval = st.tabs([
        "📖 Выдача", "🕸 Доказательства в графе", "⚖️ Контрольный метод", "📊 Оценка и ограничения"
    ])
    with tab_main:
        for item in results:
            with st.container(border=True):
                st.subheader(f'{item["rank"]}. {item["title"]}')
                st.caption(f'ISBN {item["isbn"]} · косинусное сходство: {item["base_score"]:.4f}' +
                           (f' · объединённый скоринг: {item["rank_score"]:.4f}' if mode == "rerank" else ""))
                if item["explanation"]:
                    st.write(item["explanation"])
                    st.caption(f'Длина: {item["path"]["length"]} ребра; '
                               f'информативность: {item["informativeness"]:.3f}')
                else:
                    st.info("Подтверждённый объясняющий путь в заданных пределах поиска не найден.")

    with tab_graph:
        explained = [r for r in results if r["path"]]
        if not explained:
            st.info("В выданных рекомендациях нет найденных объясняющих путей.")
        else:
            chosen = st.selectbox("Рекомендация для визуализации",
                                  [r["book_id"] for r in explained],
                                  format_func=lambda bid: next(r["title"] for r in explained if r["book_id"] == bid))
            selected = next(r for r in explained if r["book_id"] == chosen)
            plot_info = path_chart_data(assets.graph, selected["path"])
            st.plotly_chart(draw_evidence_path(plot_info), width="stretch")
            st.write(selected["explanation"])
            st.caption("Линии соответствуют только фактически присутствующим рёбрам графа; "
                       "их исходные направления указаны ниже.")
            st.dataframe(pd.DataFrame(plot_info["edges"]).rename(columns={
                "from_id": "Откуда", "to_id": "Куда", "relation": "Тип связи",
                "source_file": "Файл-источник", "source_row": "Номер записи",
                "source_url": "Внешняя ссылка"
            }), hide_index=True, width="stretch")

    with tab_baseline:
        st.markdown("**Сравнение по метаданным (Jaccard)** — общие автор, издательство и жанр. "
                    "Это контрольный алгоритм, не замена Node2Vec.")
        baseline = compare_metadata_baseline(assets.graph, source, top_k=top_k)
        if baseline:
            st.dataframe(pd.DataFrame(baseline)[["rank", "title", "jaccard"]]
                         .rename(columns={"rank": "Место", "title": "Книга", "jaccard": "Jaccard"}),
                         hide_index=True, width="stretch")
        else:
            st.info("Недостаточно метаданных для контрольного метода.")

    with tab_eval:
        st.warning("Здесь показана оценка сценария пользователь → книга (holdout). "
                   "Она НЕ является оценкой точности рекомендаций книга → книга.")
        report = assets.report
        stats = report.get("preparation", {})
        metrics = report.get("user_to_book_holdout", {})
        st.write("**Состав данных:**")
        st.json({key: stats.get(key) for key in
                 ("validated_rating_rows", "train_rows", "test_rows", "kg_nodes",
                  "kg_edges", "relation_types", "genre_source", "warning")})
        st.write("**Метрики user-to-book:**")
        st.json(metrics)
        st.write("**Косвенная оценка book-to-book по отложенным предпочтениям:**")
        st.caption("Положительные тестовые книги сравниваются с рекомендациями по ранее "
                   "понравившимся книгам. Это не экспертная оценка сходства пар книг.")
        proxy = report.get("book_to_book_proxy_holdout", {})
        if proxy:
            st.json(proxy)
            methods = proxy.get("methods", {})
            if methods:
                plot_rows = [{"Метод": name, "Precision@10": vals.get("p@10", 0.0),
                              "NDCG@10": vals.get("ndcg@10", 0.0)}
                             for name, vals in methods.items()]
                st.bar_chart(pd.DataFrame(plot_rows).set_index("Метод"))
        else:
            st.info("Эксперимент book-to-book ещё не запускался.")
        st.write("**Экспериментальные параметры:**")
        st.json(report.get("parameters", {}))
        st.info("Путь доказывает наличие связи в графе, но не причинное влияние "
                "этой связи на оценку Node2Vec. Для faithfulness нужны отдельные "
                "абляционные эксперименты с изменением графа и переобучением.")
    st.caption("База данных: Book-Crossing; внешние факты жанров допускаются "
               "только с указанием происхождения. Прототип предназначен для исследования.")


if __name__ == "__main__":
    render()
