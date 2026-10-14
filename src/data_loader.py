"""
Загрузка и фильтрация данных Book-Crossing.
Строит nodes.csv и edges.csv в data/raw/, граф NetworkX в data/processed/.
"""

import os
import pandas as pd
import networkx as nx
from config import RAW_DATA_DIR, PROCESSED_DIR, TOP_BOOKS, TOP_USERS


# ─────────────────────────────── RAW CSV → nodes/edges ────────────────────────

def convert_bx_to_graph() -> bool:
    nodes_path = os.path.join(RAW_DATA_DIR, "nodes.csv")
    edges_path = os.path.join(RAW_DATA_DIR, "edges.csv")

    if os.path.exists(nodes_path) and os.path.exists(edges_path):
        print("[data_loader] nodes.csv / edges.csv уже существуют, пропускаем конвертацию.")
        return True

    print("[data_loader] Конвертация Book-Crossing CSV …")

    def _read(fname, **kw):
        path = os.path.join(RAW_DATA_DIR, fname)
        for enc in ("latin-1", "utf-8"):
            try:
                return pd.read_csv(path, sep=";", encoding=enc, on_bad_lines="skip",
                                   low_memory=False, **kw)
            except Exception:
                pass
        raise FileNotFoundError(f"Не удалось прочитать {path}")

    books   = _read("BX-Books.csv")
    ratings = _read("BX-Book-Ratings.csv")
    print(f"  Загружено: {len(books)} книг, {len(ratings)} оценок")

    # ── фильтрация ─────────────────────────────────────────────────────────────
    rated          = ratings[ratings["Book-Rating"] > 0]
    top_isbns      = rated["ISBN"].value_counts().head(TOP_BOOKS).index
    top_user_ids   = rated["User-ID"].value_counts().head(TOP_USERS).index

    fbooks   = books[books["ISBN"].isin(top_isbns)].copy()
    fratings = rated[rated["ISBN"].isin(top_isbns) & rated["User-ID"].isin(top_user_ids)].copy()
    print(f"  После фильтра: {len(fbooks)} книг, {len(fratings)} оценок")

    # ── узлы ──────────────────────────────────────────────────────────────────
    nodes = []
    for _, r in fbooks.iterrows():
        if pd.notna(r["ISBN"]) and pd.notna(r["Book-Title"]):
            nodes.append({"id": f"book_{r['ISBN']}", "type": "book",
                          "label": str(r["Book-Title"])[:100],
                          "isbn": str(r["ISBN"])})
    for auth in fbooks["Book-Author"].dropna().unique():
        nodes.append({"id": f"author_{auth}", "type": "author",
                      "label": str(auth)[:80], "isbn": ""})
    for uid in top_user_ids:
        nodes.append({"id": f"user_{uid}", "type": "user",
                      "label": f"User {uid}", "isbn": ""})

    nodes_df = pd.DataFrame(nodes).drop_duplicates("id").dropna(subset=["id"])
    print(f"  Узлов: {len(nodes_df)}  {nodes_df['type'].value_counts().to_dict()}")

    # ── рёбра ─────────────────────────────────────────────────────────────────
    edges = []
    for _, r in fbooks.iterrows():
        if pd.notna(r["ISBN"]) and pd.notna(r["Book-Author"]):
            edges.append({"source": f"author_{r['Book-Author']}",
                          "target": f"book_{r['ISBN']}",
                          "relation": "wrote", "weight": 1.0,
                          "rating": 0})
    for _, r in fratings.iterrows():
        if pd.notna(r["ISBN"]) and pd.notna(r["User-ID"]):
            edges.append({"source": f"user_{r['User-ID']}",
                          "target": f"book_{r['ISBN']}",
                          "relation": "rated",
                          "weight": max(0.1, float(r["Book-Rating"]) / 10.0),
                          "rating": int(r["Book-Rating"])})

    edges_df = pd.DataFrame(edges).dropna(subset=["source", "target"])
    print(f"  Рёбер: {len(edges_df)}")

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    print(f"  Сохранено: {nodes_path}\n           {edges_path}")
    return True


# ─────────────────────────────── graph helpers ────────────────────────────────

def load_nodes() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DATA_DIR, "nodes.csv"))


def load_edges() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DATA_DIR, "edges.csv"))


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["id"], type=r["type"], label=r["label"])
    for _, r in edges_df.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(r.get("weight", 1.0)))
    return G


def save_graph(G: nx.Graph, fname="graph.graphml"):
    nx.write_graphml(G, os.path.join(PROCESSED_DIR, fname))


def load_graph(fname="graph.graphml") -> nx.Graph:
    return nx.read_graphml(os.path.join(PROCESSED_DIR, fname))


def get_or_build_graph() -> nx.Graph:
    gpath     = os.path.join(PROCESSED_DIR, "graph.graphml")
    nodes_path = os.path.join(RAW_DATA_DIR, "nodes.csv")
    edges_path = os.path.join(RAW_DATA_DIR, "edges.csv")

    # кэш валиден только если все три файла на месте
    if os.path.exists(gpath) and os.path.exists(nodes_path) and os.path.exists(edges_path):
        print("[data_loader] Загружаем кэшированный граф …")
        return load_graph()

    # иначе — конвертируем с нуля
    convert_bx_to_graph()
    G = build_graph(load_nodes(), load_edges())
    save_graph(G)
    print(f"[data_loader] Граф сохранён: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    return G


if __name__ == "__main__":
    G = get_or_build_graph()
    print(f"Готово: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
