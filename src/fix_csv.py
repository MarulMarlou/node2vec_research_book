import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from config import METRICS_DIR, EMBEDDINGS_DIR
from data_loader import load_edges
from metrics import compute_all_metrics
import node2vec_runner as n2v
import transe_runner   as transe
from visualizer import generate_all_plots

edges_df = load_edges()

CSV_COLUMNS = [
    "algorithm", "experiment_id", "params",
    "p@5",  "p@10",  "p@20",
    "r@5",  "r@10",  "r@20",
    "f1@5", "f1@10", "f1@20",
    "ndcg@5","ndcg@10","ndcg@20",
    "mrr", "auc_roc",
    "train_time", "inference_time",
]

# ── параметры Node2Vec (должны совпадать с run_experiments.py) ─────────────────
NODE2VEC_EXPERIMENTS = [
    {"experiment_id": "n2v_01", "p": 1.0, "q": 1.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_02", "p": 0.5, "q": 1.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_03", "p": 2.0, "q": 1.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_04", "p": 4.0, "q": 1.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_05", "p": 1.0, "q": 0.25, "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_06", "p": 1.0, "q": 0.5,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_07", "p": 1.0, "q": 2.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_08", "p": 1.0, "q": 4.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_09", "p": 1.0, "q": 1.0,  "walk_length": 40,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_10", "p": 1.0, "q": 1.0,  "walk_length": 80,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_11", "p": 1.0, "q": 1.0,  "walk_length": 100, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_12", "p": 1.0, "q": 1.0,  "walk_length": 60,  "num_walks": 8,  "dimensions": 128},
    {"experiment_id": "n2v_13", "p": 1.0, "q": 1.0,  "walk_length": 60,  "num_walks": 20, "dimensions": 128},
    {"experiment_id": "n2v_14", "p": 1.0, "q": 1.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 64},
    {"experiment_id": "n2v_15", "p": 1.0, "q": 1.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 256},
    {"experiment_id": "n2v_16", "p": 4.0, "q": 0.25, "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_17", "p": 0.5, "q": 4.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_18", "p": 2.0, "q": 0.5,  "walk_length": 80,  "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_19", "p": 0.5, "q": 2.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 256},
    {"experiment_id": "n2v_20", "p": 4.0, "q": 2.0,  "walk_length": 60,  "num_walks": 15, "dimensions": 128},
]

TRANSE_EXPERIMENTS = [
    {"experiment_id": "transe_01", "embedding_dim": 64,  "learning_rate": 0.01,  "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_02", "embedding_dim": 64,  "learning_rate": 0.01,  "loss": "marginranking",   "margin": 2.0},
    {"experiment_id": "transe_03", "embedding_dim": 64,  "learning_rate": 0.001, "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_04", "embedding_dim": 64,  "learning_rate": 0.1,   "loss": "bceaftersigmoid", "margin": 1.0},
    {"experiment_id": "transe_05", "embedding_dim": 128, "learning_rate": 0.01,  "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_06", "embedding_dim": 128, "learning_rate": 0.01,  "loss": "marginranking",   "margin": 2.0},
    {"experiment_id": "transe_07", "embedding_dim": 128, "learning_rate": 0.001, "loss": "bceaftersigmoid", "margin": 1.0},
    {"experiment_id": "transe_08", "embedding_dim": 128, "learning_rate": 0.1,   "loss": "marginranking",   "margin": 2.0},
    {"experiment_id": "transe_09", "embedding_dim": 256, "learning_rate": 0.01,  "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_10", "embedding_dim": 256, "learning_rate": 0.001, "loss": "bceaftersigmoid", "margin": 2.0},
]

rows = []

# ── Node2Vec ───────────────────────────────────────────────────────────────────
print("Пересчёт метрик Node2Vec...")
for params in NODE2VEC_EXPERIMENTS:
    exp_id = params["experiment_id"]
    npy = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    if not os.path.exists(npy):
        print(f"  {exp_id}: эмбеддинги не найдены, пропускаем")
        continue
    try:
        emb_df  = n2v.load_embeddings(exp_id)
        metrics = compute_all_metrics(emb_df, edges_df)
        row = {
            "algorithm":      "Node2Vec",
            "experiment_id":  exp_id,
            "params":         json.dumps({k: params[k] for k in
                              ("p","q","walk_length","num_walks","dimensions")}),
            **metrics,
            "train_time":     0,
            "inference_time": 0,
        }
        rows.append({c: row.get(c, 0.0) for c in CSV_COLUMNS})
        print(f"  {exp_id}: P@10={metrics.get('p@10',0):.4f}  "
              f"NDCG@10={metrics.get('ndcg@10',0):.4f}  "
              f"MRR={metrics.get('mrr',0):.4f}")
    except Exception as e:
        print(f"  {exp_id}: ОШИБКА — {e}")

# ── TransE ─────────────────────────────────────────────────────────────────────
print("\nПересчёт метрик TransE...")
for params in TRANSE_EXPERIMENTS:
    exp_id = params["experiment_id"]
    npy = os.path.join(EMBEDDINGS_DIR, f"{exp_id}_emb.npy")
    if not os.path.exists(npy):
        print(f"  {exp_id}: эмбеддинги не найдены, пропускаем")
        continue
    try:
        emb_df, rated_vec = transe.load_embeddings(exp_id)
        emb_adj = transe.build_transe_emb_df_for_metrics(emb_df, rated_vec)
        metrics = compute_all_metrics(emb_adj, edges_df)
        row = {
            "algorithm":      "TransE",
            "experiment_id":  exp_id,
            "params":         json.dumps({k: params[k] for k in
                              ("embedding_dim","learning_rate","loss","margin")}),
            **metrics,
            "train_time":     0,
            "inference_time": 0,
        }
        rows.append({c: row.get(c, 0.0) for c in CSV_COLUMNS})
        print(f"  {exp_id}: P@10={metrics.get('p@10',0):.4f}  "
              f"NDCG@10={metrics.get('ndcg@10',0):.4f}  "
              f"MRR={metrics.get('mrr',0):.4f}")
    except Exception as e:
        print(f"  {exp_id}: ОШИБКА — {e}")

# ── сохраняем ─────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows, columns=CSV_COLUMNS)
dim_map = {}
for p in NODE2VEC_EXPERIMENTS:
    dim_map[p["experiment_id"]] = p["dimensions"]
for p in TRANSE_EXPERIMENTS:
    dim_map[p["experiment_id"]] = p["embedding_dim"]

df["dimension_size"] = df["experiment_id"].map(dim_map)
CSV = os.path.join(METRICS_DIR, "all_experiments.csv")
df.to_csv(CSV, index=False)
print(f"\nСохранено {len(df)} строк → {CSV}")
print(df[["algorithm","experiment_id","p@10","ndcg@10","mrr"]].to_string())

# ── графики ───────────────────────────────────────────────────────────────────
n2v_df = df[df["algorithm"] == "Node2Vec"]
t_df   = df[df["algorithm"] == "TransE"]

best_n2v_emb = None
if len(n2v_df):
    best_n2v_id  = n2v_df.loc[n2v_df["p@10"].idxmax(), "experiment_id"]
    best_n2v_emb = n2v.load_embeddings(best_n2v_id)
    print(f"\nЛучший Node2Vec: {best_n2v_id}")

best_t_emb = None
if len(t_df):
    best_t_id    = t_df.loc[t_df["p@10"].idxmax(), "experiment_id"]
    emb, rv      = transe.load_embeddings(best_t_id)
    best_t_emb   = transe.build_transe_emb_df_for_metrics(emb, rv)
    print(f"Лучший TransE:  {best_t_id}")

generate_all_plots(df, best_n2v_emb, best_t_emb)
print("\nГотово!")
