"""
Все метрики качества рекомендаций:
  Precision@k, Recall@k, F1@k, NDCG@k, MRR, AUC-ROC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────── вспомогательные ──────────────────────────────────

def _user_book_split(emb_df: pd.DataFrame):
    users = [n for n in emb_df.index if str(n).startswith("user_")]
    books = [n for n in emb_df.index if str(n).startswith("book_")]
    return users, books


def _true_books_map(edges_df: pd.DataFrame) -> dict:
    """user_id → set of book_ids (все связи rated)"""
    m: dict = {}
    for _, r in edges_df.iterrows():
        s, t = str(r["source"]), str(r["target"])
        if s.startswith("user_") and t.startswith("book_"):
            m.setdefault(s, set()).add(t)
    return m


def _relevant_map(edges_df: pd.DataFrame, threshold: int = 7) -> dict:
    """user_id → set of book_ids с оценкой ≥ threshold (для NDCG/MRR/AUC)"""
    m: dict = {}
    for _, r in edges_df.iterrows():
        s, t = str(r["source"]), str(r["target"])
        if s.startswith("user_") and t.startswith("book_"):
            rating = int(r.get("rating", 0)) if pd.notna(r.get("rating", 0)) else 0
            if rating >= threshold:
                m.setdefault(s, set()).add(t)
    return m


def _topk_indices(user_emb: np.ndarray, book_emb: np.ndarray, k: int) -> np.ndarray:
    sim = cosine_similarity(user_emb, book_emb)       # (U, B)
    return np.argsort(-sim, axis=1)[:, :k]


# ─────────────────────────── основная функция ─────────────────────────────────

def compute_all_metrics(emb_df: pd.DataFrame,
                        edges_df: pd.DataFrame,
                        ks=(5, 10, 20),
                        test_ratio: float = 0.2,
                        random_state: int = 42) -> dict:
    """
    Для каждого пользователя откладывает test_ratio рёбер в тест,
    остальные считаются известными. Метрики считаются только по тест-рёбрам.
    """
    import random as _rnd
    _rnd.seed(random_state)
    np.random.seed(random_state)

    users, books = _user_book_split(emb_df)
    if not users or not books:
        return {}

    user_emb = emb_df.loc[users].values.astype(float)
    book_emb = emb_df.loc[books].values.astype(float)
    book_idx = {b: i for i, b in enumerate(books)}

    # строим карты: все рёбра и релевантные (rating >= 7)
    all_map = _true_books_map(edges_df)
    rel_map = _relevant_map(edges_df)

    # ── train/test split per user ──────────────────────────────────────────
    test_map = {}
    for u, book_set in all_map.items():
        book_list = list(book_set)
        _rnd.shuffle(book_list)
        n_test = max(1, int(len(book_list) * test_ratio))
        test_map[u] = set(book_list[:n_test])   # только это используем для оценки

    result = {}

    for k in ks:
        topk_idx = _topk_indices(user_emb, book_emb, k)

        p_list, r_list, f1_list, ndcg_list = [], [], [], []

        for i, u in enumerate(users):
            test_set = test_map.get(u, set())
            if not test_set:
                continue

            rec = [books[j] for j in topk_idx[i]]

            hits = len(set(rec) & test_set)
            prec = hits / k
            rec_val = hits / len(test_set)
            p_list.append(prec)
            r_list.append(rec_val)

            f1 = (2 * prec * rec_val / (prec + rec_val)) if (prec + rec_val) > 0 else 0.0
            f1_list.append(f1)

            # NDCG: релевантность = оценка >= 7
            rel_set = rel_map.get(u, set()) & test_set
            dcg = sum(1 / np.log2(rank + 2)
                      for rank, b in enumerate(rec) if b in rel_set)
            ideal = min(len(rel_set), k)
            idcg = sum(1 / np.log2(r + 2) for r in range(ideal))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        n = len(p_list) or 1
        result[f"p@{k}"]    = round(sum(p_list)    / n, 4)
        result[f"r@{k}"]    = round(sum(r_list)    / n, 4)
        result[f"f1@{k}"]   = round(sum(f1_list)   / n, 4)
        result[f"ndcg@{k}"] = round(sum(ndcg_list) / n, 4)

    # ── MRR ───────────────────────────────────────────────────────────────
    max_k = max(ks)
    topk_mrr = _topk_indices(user_emb, book_emb, max_k)
    rr_list = []
    for i, u in enumerate(users):
        test_set = test_map.get(u, set())
        if not test_set:
            continue
        rr = 0.0
        for rank, j in enumerate(topk_mrr[i]):
            if books[j] in test_set:
                rr = 1.0 / (rank + 1)
                break
        rr_list.append(rr)
    result["mrr"] = round(sum(rr_list) / len(rr_list), 4) if rr_list else 0.0

    # ── AUC-ROC ───────────────────────────────────────────────────────────
    sim_full = cosine_similarity(user_emb, book_emb)
    auc_list = []
    for i, u in enumerate(users):
        test_set = test_map.get(u, set())
        if not test_set:
            continue
        y_true = np.array([1 if b in test_set else 0 for b in books])
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        try:
            auc_list.append(roc_auc_score(y_true, sim_full[i]))
        except Exception:
            pass
    result["auc_roc"] = round(sum(auc_list) / len(auc_list), 4) if auc_list else 0.0

    return result
