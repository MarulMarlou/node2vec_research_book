import os

# config.py находится в src/, поэтому поднимаемся на уровень выше — в корень проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    EMBEDDINGS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
]:
    os.makedirs(path, exist_ok=True)

TOP_BOOKS = 5000
TOP_USERS = 1000

N2V_BASELINE = {
    "dimensions": 128,
    "walk_length": 80,
    "num_walks": 20,
}

TRANSE_NUM_EPOCHS = 50
