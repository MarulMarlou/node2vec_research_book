"""
Точка входа — запускает все 30 экспериментов:
  • 20 × Node2Vec
  •  10 × TransE
Сохраняет results/metrics/all_experiments.csv и все графики.

Запуск:
    python run_experiments.py
"""

import os
import json
import time
import numpy as np
import pandas as pd

from config import METRICS_DIR, EMBEDDINGS_DIR, PLOTS_DIR
from data_loader import get_or_build_graph, load_edges
import node2vec_runner as n2v
import transe_runner   as transe
from metrics   import compute_all_metrics
from visualizer import generate_all_plots
import datetime as dt


# ═══════════════════════════════════════════════════════════════════════════════
#  20 конфигураций Node2Vec
#  Стратегия: baseline (p=1, q=1, wl=80, nw=20, dim=128) +
#  варьирование по одному параметру за раз.
# ═══════════════════════════════════════════════════════════════════════════════
NODE2VEC_EXPERIMENTS = [
    # ── baseline (уменьшенный) ──
    {"experiment_id": "n2v_01", "p": 1.0, "q": 1.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},

    # ── варьируем p (q=1, wl=60, nw=15, dim=128) ──
    {"experiment_id": "n2v_02", "p": 0.5, "q": 1.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_03", "p": 2.0, "q": 1.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_04", "p": 4.0, "q": 1.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},

    # ── варьируем q (p=1, wl=60, nw=15, dim=128) ──
    {"experiment_id": "n2v_05", "p": 1.0, "q": 0.25,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_06", "p": 1.0, "q": 0.5,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_07", "p": 1.0, "q": 2.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_08", "p": 1.0, "q": 4.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},

    # ── варьируем walk_length (p=1, q=1, nw=15, dim=128) ──
    {"experiment_id": "n2v_09", "p": 1.0, "q": 1.0,
     "walk_length": 40, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_10", "p": 1.0, "q": 1.0,
     "walk_length": 80, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_11", "p": 1.0, "q": 1.0,
     "walk_length": 100, "num_walks": 15, "dimensions": 128},

    # ── варьируем num_walks (p=1, q=1, wl=60, dim=128) ──
    {"experiment_id": "n2v_12", "p": 1.0, "q": 1.0,
     "walk_length": 60, "num_walks": 8, "dimensions": 128},
    {"experiment_id": "n2v_13", "p": 1.0, "q": 1.0,
     "walk_length": 60, "num_walks": 20, "dimensions": 128},

    # ── var dimensions (p=1, q=1, wl=60, nw=15) ──
    {"experiment_id": "n2v_14", "p": 1.0, "q": 1.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 64},
    {"experiment_id": "n2v_15", "p": 1.0, "q": 1.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 256},

    # ── экстремальные p/q ──
    {"experiment_id": "n2v_16", "p": 4.0, "q": 0.25,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_17", "p": 0.5, "q": 4.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},

    # ── комбинированные ──
    {"experiment_id": "n2v_18", "p": 2.0, "q": 0.5,
     "walk_length": 80, "num_walks": 15, "dimensions": 128},
    {"experiment_id": "n2v_19", "p": 0.5, "q": 2.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 256},
    {"experiment_id": "n2v_20", "p": 4.0, "q": 2.0,
     "walk_length": 60, "num_walks": 15, "dimensions": 128},
]

# ═══════════════════════════════════════════════════════════════════════════════
#  10 конфигураций TransE
# ═══════════════════════════════════════════════════════════════════════════════
TRANSE_EXPERIMENTS = [
    {"experiment_id": "transe_01", "embedding_dim": 64,  "learning_rate": 0.01,  "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_02", "embedding_dim": 64,  "learning_rate": 0.01,  "loss": "marginranking",   "margin": 2.0},
    {"experiment_id": "transe_03", "embedding_dim": 64,  "learning_rate": 0.001, "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_04", "embedding_dim": 64,  "learning_rate": 0.1,   "loss": "bceaftersigmoid", "margin": 1.0},
    {"experiment_id": "transe_05", "embedding_dim": 128, "learning_rate": 0.01,  "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_06", "embedding_dim": 128, "learning_rate": 0.01,  "loss": "marginranking",   "margin": 2.0},
    {"experiment_id": "transe_07", "embedding_dim": 128, "learning_rate": 0.001, "loss": "bceaftersigmoid", "margin": 1.0},
    {"experiment_id": "transe_08", "experiment_id": "transe_08", "embedding_dim": 128, "learning_rate": 0.1,   "loss": "marginranking",   "margin": 2.0},
    {"experiment_id": "transe_09", "embedding_dim": 256, "learning_rate": 0.01,  "loss": "marginranking",   "margin": 1.0},
    {"experiment_id": "transe_10", "embedding_dim": 256, "learning_rate": 0.001, "loss": "bceaftersigmoid", "margin": 2.0},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV schema
# ═══════════════════════════════════════════════════════════════════════════════
CSV_COLUMNS = [
    "algorithm", "experiment_id", "params",
    "p@5",  "p@10",  "p@20",
    "r@5",  "r@10",  "r@20",
    "f1@5", "f1@10", "f1@20",
    "ndcg@5","ndcg@10","ndcg@20",
    "mrr", "auc_roc",
    "train_time", "inference_time",
]

ALL_EXPERIMENTS_CSV = os.path.join(METRICS_DIR, "all_experiments.csv")


def _load_existing() -> list:
    if os.path.exists(ALL_EXPERIMENTS_CSV):
        df = pd.read_csv(ALL_EXPERIMENTS_CSV)
        return df["experiment_id"].tolist()
    return []


def _append_row(row: dict):
    clean = {c: row.get(c, 0.0) for c in CSV_COLUMNS}
    df = pd.DataFrame([clean])
    if os.path.exists(ALL_EXPERIMENTS_CSV):
        df.to_csv(ALL_EXPERIMENTS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(ALL_EXPERIMENTS_CSV, index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Node2Vec experiments
# ═══════════════════════════════════════════════════════════════════════════════

def run_node2vec_experiments(G, edges_df, done_ids, total_count, counter):
    best_score, best_emb, best_id = -1, None, None
    start_global = dt.datetime.now()

    for pos, params in enumerate(NODE2VEC_EXPERIMENTS, start=1):
        exp_id = params["experiment_id"]
        counter[0] += 1
        idx = counter[0]

        # ETA по Node2Vec
        elapsed = (dt.datetime.now() - start_global).total_seconds()
        avg = elapsed / pos
        remaining = avg * (len(NODE2VEC_EXPERIMENTS) - pos)
        eta = (dt.datetime.now() + dt.timedelta(seconds=remaining)).strftime("%H:%M")

        print(f"[ETA Node2Vec] прошло ~{elapsed/60:.1f} мин, "
              f"среднее {avg/60:.1f} мин/эксп, осталось ≈{remaining/60:.1f} мин "
              f"(до ~{eta})")

        if exp_id in done_ids:
            print(f"[Exp {idx}/{total_count}] {exp_id} — пропускаем (уже выполнен)")
            # загружаем для best
            try:
                emb_df = n2v.load_embeddings(exp_id)
                metrics = compute_all_metrics(emb_df, edges_df)
                score = metrics.get("p@10", 0)
                if score > best_score:
                    best_score, best_emb, best_id = score, emb_df, exp_id
            except Exception:
                pass
            continue

        print(f"\n[Exp {idx}/{total_count}] Node2Vec  "
              f"p={params['p']} q={params['q']} "
              f"wl={params['walk_length']} nw={params['num_walks']} "
              f"dim={params['dimensions']} …", flush=True)

        try:
            t_inf0 = time.time()
            emb_df, train_time = n2v.run_node2vec(G, params)
            n2v.save_embeddings(emb_df, exp_id)

            metrics       = compute_all_metrics(emb_df, edges_df)
            inference_time = time.time() - t_inf0 - train_time

            row = {
                "algorithm":      "Node2Vec",
                "experiment_id":  exp_id,
                "params":         json.dumps({k: params[k] for k in
                                              ("p","q","walk_length","num_walks","dimensions")}),
                **{c: metrics.get(c, 0.0) for c in CSV_COLUMNS
                   if c not in ("algorithm","experiment_id","params","train_time","inference_time")},
                "train_time":     round(train_time, 2),
                "inference_time": round(inference_time, 4),
                # extra columns for convenience
                "dimensions":     params["dimensions"],
            }
            # fill missing metric columns with 0
            for col in CSV_COLUMNS:
                row.setdefault(col, 0.0)

            _append_row(row)
            done_ids.append(exp_id)

            score = metrics.get("p@10", 0)
            if score > best_score:
                best_score, best_emb, best_id = score, emb_df, exp_id

            print(f"  done in {train_time:.0f}s | P@10={metrics.get('p@10',0):.4f} "
                  f"NDCG@10={metrics.get('ndcg@10',0):.4f} MRR={metrics.get('mrr',0):.4f}")

        except Exception as e:
            print(f"  ОШИБКА: {e}")

    return best_emb, best_id


# ═══════════════════════════════════════════════════════════════════════════════
#  TransE experiments
# ═══════════════════════════════════════════════════════════════════════════════

def run_transe_experiments(edges_df, done_ids, total_count, counter):
    best_score, best_emb_adj, best_id = -1, None, None
    start_global = dt.datetime.now()

    for pos, params in enumerate(TRANSE_EXPERIMENTS, start=1):
        exp_id = params["experiment_id"]
        counter[0] += 1
        idx = counter[0]

        elapsed = (dt.datetime.now() - start_global).total_seconds()
        avg = elapsed / pos
        remaining = avg * (len(TRANSE_EXPERIMENTS) - pos)
        eta = (dt.datetime.now() + dt.timedelta(seconds=remaining)).strftime("%H:%M")

        print(f"[ETA TransE] прошло ~{elapsed/60:.1f} мин, "
              f"среднее {avg/60:.1f} мин/эксп, осталось ≈{remaining/60:.1f} мин "
              f"(до ~{eta})")

        if exp_id in done_ids:
            print(f"[Exp {idx}/{total_count}] {exp_id} — пропускаем (уже выполнен)")
            try:
                emb_df, rated_vec = transe.load_embeddings(exp_id)
                emb_adj = transe.build_transe_emb_df_for_metrics(emb_df, rated_vec)
                metrics = compute_all_metrics(emb_adj, edges_df)
                score   = metrics.get("p@10", 0)
                if score > best_score:
                    best_score, best_emb_adj, best_id = score, emb_adj, exp_id
            except Exception:
                pass
            continue

        print(f"\n[Exp {idx}/{total_count}] TransE  "
              f"dim={params['embedding_dim']} lr={params['learning_rate']} "
              f"loss={params['loss']} margin={params['margin']} …", flush=True)

        try:
            emb_df, rated_vec, train_time, inference_time = transe.run_transe(
                edges_df, params)
            transe.save_embeddings(emb_df, rated_vec, exp_id)

            emb_adj = transe.build_transe_emb_df_for_metrics(emb_df, rated_vec)
            metrics = compute_all_metrics(emb_adj, edges_df)

            row = {
                "algorithm":      "TransE",
                "experiment_id":  exp_id,
                "params":         json.dumps({k: params[k] for k in
                                              ("embedding_dim","learning_rate","loss","margin")}),
                **{c: metrics.get(c, 0.0) for c in CSV_COLUMNS
                   if c not in ("algorithm","experiment_id","params","train_time","inference_time")},
                "train_time":     round(train_time, 2),
                "inference_time": round(inference_time, 4),
                "dimensions":     params["embedding_dim"],
            }
            for col in CSV_COLUMNS:
                row.setdefault(col, 0.0)

            _append_row(row)
            done_ids.append(exp_id)

            score = metrics.get("p@10", 0)
            if score > best_score:
                best_score, best_emb_adj, best_id = score, emb_adj, exp_id

            print(f"  done in {train_time:.0f}s | P@10={metrics.get('p@10',0):.4f} "
                  f"NDCG@10={metrics.get('ndcg@10',0):.4f} MRR={metrics.get('mrr',0):.4f}")

        except Exception as e:
            print(f"  ОШИБКА: {e}")

    return best_emb_adj, best_id


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ЗАПУСК 30 ЭКСПЕРИМЕНТОВ (Node2Vec + TransE)")
    print("=" * 60)

    G        = get_or_build_graph()
    edges_df = load_edges()

    done_ids   = _load_existing()
    total      = len(NODE2VEC_EXPERIMENTS) + len(TRANSE_EXPERIMENTS)
    counter    = [0]  # mutable counter passed by ref

    # ── Node2Vec ──────────────────────────────────────────────────────────────
    best_n2v_emb, best_n2v_id = run_node2vec_experiments(
        G, edges_df, done_ids, total, counter)

    # ── TransE ────────────────────────────────────────────────────────────────
    best_transe_emb, best_transe_id = run_transe_experiments(
        edges_df, done_ids, total, counter)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if os.path.exists(ALL_EXPERIMENTS_CSV):
        results_df = pd.read_csv(ALL_EXPERIMENTS_CSV)
        from visualizer import generate_all_plots
        generate_all_plots(
            results_df,
            best_n2v_emb   if best_n2v_emb    is not None else None,
            best_transe_emb if best_transe_emb is not None else None,
        )

    print("\n" + "=" * 60)
    print("  ГОТОВО! Результаты сохранены:")
    print(f"    {ALL_EXPERIMENTS_CSV}")
    print(f"    {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
