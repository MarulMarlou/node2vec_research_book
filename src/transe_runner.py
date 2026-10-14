"""
TransE через PyKEEN.
Тройки: (user_id, "rated", isbn)  и  (author_name, "wrote", isbn)
Скоринг: score(u,b) = -|| h_u + r_rated - t_b ||_2
"""

import os
import time
import numpy as np
import pandas as pd

from config import EMBEDDINGS_DIR, RAW_DATA_DIR, TRANSE_NUM_EPOCHS

try:
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    _PYKEEN = True
except ImportError:
    _PYKEEN = False
    print("[transe_runner] PyKEEN не установлен: pip install pykeen torch")


# ─────────────────────────── helpers ──────────────────────────────────────────

def _build_triples(edges_df: pd.DataFrame) -> np.ndarray:
    """
    Строит массив троек (head, relation, tail) из edges.csv.
    """
    rows = []
    for _, r in edges_df.iterrows():
        s, t, rel = str(r["source"]), str(r["target"]), str(r.get("relation", ""))
        if rel in ("rated", "wrote"):
            rows.append((s, rel, t))
        elif s.startswith("user_") and t.startswith("book_"):
            rows.append((s, "rated", t))
        elif s.startswith("author_") and t.startswith("book_"):
            rows.append((s, "wrote", t))
    return np.array(rows, dtype=str)


# ─────────────────────────── public API ───────────────────────────────────────

def run_transe(edges_df: pd.DataFrame, params: dict) -> tuple:
    """
    Обучает TransE и возвращает (emb_df, train_time, inference_time).
    """
    if not _PYKEEN:
        raise RuntimeError("PyKEEN не установлен.")

    dim        = params["embedding_dim"]
    lr         = params["learning_rate"]
    loss       = params["loss"]
    margin     = params["margin"]
    num_epochs = params.get("num_epochs", TRANSE_NUM_EPOCHS)

    triples_np = _build_triples(edges_df)
    if len(triples_np) == 0:
        raise ValueError("Нет троек для обучения TransE.")

    tf = TriplesFactory.from_labeled_triples(triples_np)

    # Разбивка 80/10/10
    train_tf, valid_tf, test_tf = tf.split([0.8, 0.1, 0.1], random_state=42)

    t0 = time.time()

    loss_map = {
        "marginranking":     "MarginRankingLoss",
        "bceaftersigmoid":   "BCEAfterSigmoidLoss",
    }
    loss_name = loss_map.get(loss.lower(), "MarginRankingLoss")

    loss_kwargs = {"margin": margin} if "Margin" in loss_name else {}

    result = pipeline(
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        model="TransE",
        model_kwargs={"embedding_dim": dim},
        loss=loss_name,
        loss_kwargs=loss_kwargs if loss_kwargs else None,
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        training_kwargs={
            "num_epochs": num_epochs,
            "batch_size": 512,
        },
        random_seed=42,
        use_tqdm=False,
    )
    train_time = time.time() - t0

    t1 = time.time()
    model = result.model
    model.eval()

    entity_repr = tf.entity_to_id          # dict str → int
    with torch.no_grad():
        all_embs = model.entity_representations[0](
            indices=None
        ).cpu().numpy()                     # (num_entities, dim)

    emb_dict = {ent: all_embs[idx] for ent, idx in entity_repr.items()}
    emb_df   = pd.DataFrame.from_dict(emb_dict, orient="index")
    emb_df.index.name = "node_id"

    # сохраняем также вектор отношения "rated" для скоринга
    rel_repr = tf.relation_to_id
    with torch.no_grad():
        all_rel_embs = model.relation_representations[0](
            indices=None
        ).cpu().numpy()
    rated_idx = rel_repr.get("rated", 0)
    rated_vec = all_rel_embs[rated_idx]

    inference_time = time.time() - t1

    return emb_df, rated_vec, train_time, inference_time


def transe_recommend(user_id: str, book_ids: list,
                     emb_df: pd.DataFrame, rated_vec: np.ndarray,
                     top_k: int = 10) -> list:
    """
    score(u,b) = -|| h_u + r_rated - t_b ||_2
    Возвращает список (book_id, score) отсортированный по убыванию.
    """
    if user_id not in emb_df.index:
        return []
    h_u = emb_df.loc[user_id].values
    scores = []
    for b in book_ids:
        if b not in emb_df.index:
            continue
        t_b = emb_df.loc[b].values
        s   = -np.linalg.norm(h_u + rated_vec - t_b)
        scores.append((b, float(s)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_transe_emb_df_for_metrics(emb_df: pd.DataFrame,
                                    rated_vec: np.ndarray) -> pd.DataFrame:
    rows = {}
    for nid in emb_df.index:
        if str(nid).startswith("user_"):
            rows[nid] = emb_df.loc[nid].values + rated_vec
        else:
            rows[nid] = emb_df.loc[nid].values
    return pd.DataFrame.from_dict(rows, orient="index")


def save_embeddings(emb_df: pd.DataFrame, rated_vec: np.ndarray, exp_id: str):
    npy  = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    idx  = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_idx.txt")
    rvec = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_rated_vec.npy")
    np.save(npy, emb_df.values)
    np.save(rvec, rated_vec)
    with open(idx, "w", encoding="utf-8") as f:
        f.write("\n".join(emb_df.index.tolist()))


def load_embeddings(exp_id: str):
    npy  = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    idx  = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_idx.txt")
    rvec_path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_rated_vec.npy")
    vals = np.load(npy)
    with open(idx, encoding="utf-8") as f:
        index = [l.strip() for l in f]
    emb_df = pd.DataFrame(vals, index=index)
    rated_vec = np.load(rvec_path) if os.path.exists(rvec_path) else np.zeros(vals.shape[1])
    return emb_df, rated_vec
