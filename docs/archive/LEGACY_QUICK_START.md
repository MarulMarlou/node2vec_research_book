> **АРХИВ (предыдущая курсовая версия).** Команды и протокол ниже не являются актуальным руководством ВКР. Для нового конвейера используйте [корневой README](../../README.md) и [инструкцию Windows](../VKR_WINDOWS.md).

#  КРАТКОЕ РУКОВОДСТВО - Node2Vec Research System

##  Быстрый старт (5 минут)

### 1. Подготовка окружения
```bash
# Создать виртуальное окружение
python -m venv .venv

# Активировать (Windows)
.venv\Scripts\activate

# Активировать (Linux/Mac)  
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

### 2. Скачать данные
1. Перейти: https://www.kaggle.com/datasets/quhujun/book-crossing
2. Скачать архив и извлечь `BX-Books.csv` и `BX-Book-Ratings.csv`
3. Поместить в папку `data/raw/`

### 3. Запустить эксперименты

**Автоматический запуск (рекомендуется):**
```bash
# Windows
run_full_experiment.bat

# Linux/Mac
chmod +x run_full_experiment.sh
./run_full_experiment.sh
```

**Ручной запуск:**
```bash
python -m src.debug_check        # Проверка
python -m src.convert           # Конвертация данных  
python -m src.data_loader       # Создание графа
python -m src.experiments_updated # Эксперименты
python -m src.create_visualizations # Визуализации
python -m src.analyze_results   # Анализ
```

##  Что получится

После выполнения всех команд у вас будет:

###  Результаты  
- `results/plots/` - 7+ графиков и визуализаций
- `results/metrics/` - таблицы с метриками и анализом
- `results/embeddings/` - векторные представления узлов

###  Исходный код
- `src/` - весь Python код системы
- Все модули готовы к работе и расширению

##  Решение проблем

** "BX-Books.csv not found"**
→ Скачайте данные с Kaggle в `data/raw/`

** "No module named 'umap'"**  
→ `pip install umap-learn`

** "Найдено 0 пользователей"**
→ Перезапустите: `python -m src.convert`

** Диагностика:**
→ `python -m src.debug_check`

##  Параметры экспериментов

Можно изменить в `src/experiments_updated.py`:

```python
EXPERIMENT_PARAMETERS = [
    {'experiment_id': 'exp_001', 'p': 1.0, 'q': 1.0, ...},
    {'experiment_id': 'exp_002', 'p': 2.0, 'q': 0.5, ...},
    # Добавьте свои эксперименты
]
```

**Ключевые параметры:**
- `p` (0.5-4.0) - возвращение к предыдущему узлу
- `q` (0.25-4.0) - баланс локального/глобального поиска  
- `dimensions` (32-256) - размерность эмбеддингов

## Поддержка

- Диагностика проблем: `python -m src.debug_check`  

---