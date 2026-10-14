import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import METRICS_DIR, EMBEDDINGS_DIR
from visualizer import generate_all_plots
import node2vec_runner as n2v
import transe_runner   as transe

CSV = os.path.join(METRICS_DIR, "all_experiments.csv")

# читаем «битый» CSV — engine=python справляется с лишними полями
df = pd.read_csv(CSV, engine="python", on_bad_lines="warn")
print(f"Загружено строк: {len(df)}")
print(df[["algorithm","experiment_id","p@10","ndcg@10","mrr"]].to_string())

# сохраняем чистую версию
df.to_csv(CSV, index=False)
print("CSV пересохранён.")

# лучший Node2Vec по p@10
n2v_df = df[df["algorithm"] == "Node2Vec"]
best_n2v_id = n2v_df.loc[n2v_df["p@10"].idxmax(), "experiment_id"]
print(f"Лучший Node2Vec: {best_n2v_id}")

# лучший TransE по p@10
t_df = df[df["algorithm"] == "TransE"]
best_t_id = t_df.loc[t_df["p@10"].idxmax(), "experiment_id"] if len(t_df) else None
print(f"Лучший TransE: {best_t_id}")

# загружаем эмбеддинги
best_n2v_emb = None
try:
    best_n2v_emb = n2v.load_embeddings(best_n2v_id)
except Exception as e:
    print(f"Не удалось загрузить Node2Vec emb: {e}")

best_t_emb = None
if best_t_id:
    try:
        emb, rated_vec = transe.load_embeddings(best_t_id)
        best_t_emb = transe.build_transe_emb_df_for_metrics(emb, rated_vec)
    except Exception as e:
        print(f"Не удалось загрузить TransE emb: {e}")

generate_all_plots(df, best_n2v_emb, best_t_emb)
print("Готово! Графики в results/plots/")