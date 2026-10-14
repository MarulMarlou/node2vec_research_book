"""
Node2Vec: генерация блужданий + Skip-gram (gensim Word2Vec).
Возвращает pd.DataFrame эмбеддингов с индексом = node_id.
"""
import os, time, random, numpy as np, pandas as pd, networkx as nx
from joblib import Parallel, delayed
from config import EMBEDDINGS_DIR

try:
    from gensim.models import Word2Vec
    _GENSIM = True
except ImportError:
    _GENSIM = False


# ─────────────────────────── walk generation ──────────────────────────────────

def _node2vec_walk(adj: dict, start, walk_length: int, p: float, q: float) -> list:
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = adj.get(cur, [])
        if not nbrs:
            break
        if len(walk) == 1:
            walk.append(random.choice(nbrs))
        else:
            prev = walk[-2]
            probs = []
            for nb in nbrs:
                if nb == prev:
                    probs.append(1.0 / p)
                elif prev in adj.get(nb, []):
                    probs.append(1.0)
                else:
                    probs.append(1.0 / q)
            s = sum(probs)
            probs = [x / s for x in probs]
            walk.append(np.random.choice(nbrs, p=probs))
    return [str(n) for n in walk]

def _walks_for_nodes(nodes: list, adj: dict, walk_length: int,
                     p: float, q: float, seed: int) -> list:
    random.seed(seed)
    np.random.seed(seed)
    return [_node2vec_walk(adj, n, walk_length, p, q) for n in nodes]

def _generate_walks(G: nx.Graph, walk_length: int, num_walks: int,
                    p: float, q: float) -> list:
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            w = _node2vec_walk(G, node, walk_length, p, q)
            if len(w) > 1:
                walks.append(w)
    return walks


# ─────────────────────────── skip-gram (fallback) ─────────────────────────────

def _skipgram_fit(walks: list, dimensions: int, window: int = 5,
                  epochs: int = 5, lr: float = 0.025) -> dict:
    """Лёгкая реализация Skip-gram без gensim (медленнее)."""
    vocab = sorted({n for w in walks for n in w})
    idx   = {n: i for i, n in enumerate(vocab)}
    V     = len(vocab)
    W     = np.random.randn(V, dimensions).astype(np.float32) * 0.1
    C     = np.random.randn(V, dimensions).astype(np.float32) * 0.1

    for ep in range(epochs):
        for walk in walks:
            for i, center in enumerate(walk):
                ci = idx[center]
                ctx_words = walk[max(0, i - window): i] + walk[i + 1: i + window + 1]
                for ctx in ctx_words:
                    oi = idx[ctx]
                    dot = float(np.dot(W[ci], C[oi]))
                    sig = 1.0 / (1.0 + np.exp(-dot))
                    err = (sig - 1.0) * lr
                    W[ci] -= err * C[oi]
                    C[oi] -= err * W[ci]

    return {v: W[idx[v]] for v in vocab}


# ─────────────────────────── public API ───────────────────────────────────────

def run_node2vec(G: nx.Graph, params: dict) -> tuple:
    p, q        = params["p"], params["q"]
    walk_len    = params["walk_length"]
    num_walks   = params["num_walks"]
    dimensions  = params["dimensions"]
    n_jobs      = max(1, os.cpu_count() - 1)   # все ядра минус одно

    # строим adj-список один раз — dict быстрее nx.neighbors в цикле
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    nodes = list(G.nodes())

    t0 = time.time()

    # параллельно: каждый поток делает один «проход» по всем узлам
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_walks_for_nodes)(
            nodes, adj, walk_len, p, q, seed=i
        )
        for i in range(num_walks)
    )
    walks = [w for batch in results for w in batch if len(w) > 1]

    if _GENSIM:
        model = Word2Vec(
            sentences=walks,
            vector_size=dimensions,
            window=5, min_count=1,
            workers=n_jobs, sg=1, epochs=5,
        )
        emb = {n: model.wv[n] for n in model.wv.index_to_key}
    else:
        raise RuntimeError("gensim не установлен: pip install gensim")

    train_time = time.time() - t0
    emb_df = pd.DataFrame.from_dict(emb, orient="index")
    emb_df.index.name = "node_id"
    return emb_df, train_time


def save_embeddings(emb_df: pd.DataFrame, exp_id: str) -> str:
    path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    np.save(path, emb_df.values)
    idx_path = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_idx.txt")
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write("\n".join(emb_df.index.tolist()))
    return path


def load_embeddings(exp_id: str) -> pd.DataFrame:
    npy = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    idx = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_idx.txt")
    vals = np.load(npy)
    with open(idx, encoding="utf-8") as f:
        index = [l.strip() for l in f]
    return pd.DataFrame(vals, index=index)
