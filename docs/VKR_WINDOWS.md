# Запуск ВКР на Windows 10/11: инструкция для получателя

Документ относится **только к ветке** `feature/vkr-stage1-typed-graph-holdout` проекта [node2vec_research_book](https://github.com/MarulMarlou/node2vec_research_book/tree/feature/vkr-stage1-typed-graph-holdout). Основная версия Python: **3.11**.

## 1. Получение кода

**Вариант А — Git:**

```powershell
git clone --branch feature/vkr-stage1-typed-graph-holdout --single-branch https://github.com/MarulMarlou/node2vec_research_book.git
cd node2vec_research_book
git branch --show-current
```

Последняя команда должна показать `feature/vkr-stage1-typed-graph-holdout`.

**Вариант Б — без Git:** перейдите на [страницу ветки](https://github.com/MarulMarlou/node2vec_research_book/tree/feature/vkr-stage1-typed-graph-holdout), выберите **Code → Download ZIP**, распакуйте архив и откройте PowerShell в полученной папке.

Если у вас уже есть клон со старой веткой:

```powershell
git status --short
git fetch origin
git switch --track origin/feature/vkr-stage1-typed-graph-holdout
```

Если локальная ветка уже создана, вместо `git switch --track` используйте `git switch feature/vkr-stage1-typed-graph-holdout`, затем `git pull --ff-only`. Перед переключением сохраните свои несохранённые изменения. Не запускайте старый `run_full_experiment.bat` — это другой контур.

## 2. Проверка Python

```powershell
py -0p
py -3.11 --version
```

Если 3.11 отсутствует, установите в PowerShell через Windows Package Manager:

```powershell
winget install --id Python.Python.3.11 --exact --source winget --accept-package-agreements --accept-source-agreements
```

После установки **откройте новый терминал** и проверьте `py -3.11 --version`. Уже имеющуюся версию Python 3.15 удалять **не требуется**.

## 3. Подготовка набора данных

Нужны только два оригинальных CSV: `BX-Books.csv` и `BX-Book-Ratings.csv` из [Book-Crossing / GroupLens](https://grouplens.org/datasets/book-crossing/). `BX-Users.csv` текущему конвейеру не требуется.

Создайте каталог:

```powershell
New-Item -ItemType Directory -Force .\data\raw | Out-Null
```

Разместите:

```text
node2vec_research_book/
├── data/
│   └── raw/
│       ├── BX-Books.csv
│       └── BX-Book-Ratings.csv
├── src/
├── scripts/
├── README.md
└── ...
```

Проверьте:

```powershell
Test-Path .\data\raw\BX-Books.csv
Test-Path .\data\raw\BX-Book-Ratings.csv
```

**Обе команды должны вывести `True`.** Не пересохраняйте CSV в Excel: данные читаются как `latin-1`, разделитель `;`, с поддержкой экранированных кавычек. Исходные пользовательские рейтинги и производные файлы исключены из Git; не публикуйте их как часть исходного кода без проверки условий набора.

## 4. Первый запуск и проверка графа

Из корня проекта:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode quick
```

При первом вызове скрипт создаст **`.venv-vkr`** на Python 3.11 и установит зависимости из `requirements-vkr-app.txt`. Первый запуск может занять некоторое время из-за загрузки библиотек.

Режим `quick` использует до 150 пользователей и 1 000 книг, затем выполняет аудит. Успех: завершающие сообщения `"audit_passed": true`, `"errors": []`, `Run succeeded`. Сравнивать длительность обучения на разных ПК без описания оборудования некорректно.

Основные файлы после этого:

| Файл | Содержимое |
|---|---|
| `data/processed/vkr_knowledge_graph.graphml` | Типизированный ориентированный граф |
| `data/processed/vkr_node2vec_graph.graphml` | Проекция для Node2Vec |
| `data/processed/vkr_train_ratings.csv`, `vkr_test_ratings.csv` | Обучающие/отложенные оценки |
| `results/embeddings/vkr_n2v.npy`, `vkr_n2v_ids.json` | Эмбеддинги + порядок узлов |
| `results/metrics/vkr_pipeline_report.json` | Параметры и метрики |
| `results/metrics/vkr_graph_audit.json` | Контроль связности, источников и утечек |

Граф не содержит отложенные test-рёбра `rated`; это проверяется в `vkr_graph_audit`.

## 5. Открыть веб-интерфейс

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app -SkipInstall
```

Откройте http://localhost:8501. Выберите книгу и получите рекомендации; попробуйте объяснения `shortest` и `informative`, `posthoc` и `rerank`. Для остановки сервера нажмите **Ctrl+C** в терминале.

**Если интерфейс сообщает об отсутствии артефактов:** сначала запустите `quick` либо `main`. Не смешивайте граф и эмбеддинги из разных запусков — Dashboard проверяет SHA-256.

## 6. Основной граф, тесты и научный эксперимент

После первого запуска зависимости уже установлены; используйте `-SkipInstall`.

```powershell
# Основной подграф: до 1000 пользователей, 5000 книг
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode main -SkipInstall

# Синтетические регрессионные и интеграционные тесты
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode tests -SkipInstall

# Научный подбор параметров по отдельной validation-выборке (3 опыта)
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode study-quick -SkipInstall

# Расширенная сетка (36 запусков Node2Vec на том же уменьшенном графе)
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode study-full -SkipInstall
```

Результаты `study-quick`/`study-full`: `results/study/embedding_experiments.csv`, `results/study/study_report.json`, граф и train/validation/test CSV. Они **не подменяют** артефакты для Streamlit.

**Предупреждение:** `quick` и `main` сохраняют результаты в одинаковых папках `data/processed` и `results/embeddings`, поэтому следующий запуск перезаписывает предыдущую основную модель и отчёт. Если нужно сравнить файлы, сначала скопируйте их в отдельную локальную папку. `study-quick` и `study-full` аналогично перезаписывают каталог `results/study` между собой.

## 7. Частые вопросы

| Сообщение | Что сделать |
|---|---|
| `No suitable Python runtime found`, `Python 3.11 required` | Установите Python 3.11 через `winget`, перезапустите терминал и проверьте `py -3.11 --version` |
| `Required Book-Crossing CSV missing` | Проверьте оба `Test-Path` из п. 3 |
| `ModuleNotFoundError` | Запустите скрипт без `-SkipInstall` для установки зависимостей; не используйте системный Python 3.15 |
| `SHA-256 mismatch` | Перезапустите `quick` или `main` целиком; нельзя вручную заменить только один из файлов |
| `port 8501 already in use` | Завершите ранее запущенный Streamlit (`Ctrl+C`) или используйте другой порт при ручном запуске |
| Медленная установка зависимостей | Первый запуск скачивает NumPy, SciPy, Streamlit и другие пакеты; следующие запуски используйте с `-SkipInstall` |

## 8. Что показано, а что ещё не завершено

- Граф, Node2Vec, метрики и аудит проверены на реальном Book-Crossing, в том числе при локальном запуске на Windows.
- Существующий контроль metadata/Jaccard **не является** обучаемой моделью TransE.
- В текущем графе **нет внешних жанров**; автоматическое жанровое обогащение и научное сравнение с TransE ещё предстоят.
- `study-full` выполняет 36 вариантов **только Node2Vec**, поэтому сам по себе не считается выполнением сравнения Node2Vec/TransE по плану стажировки.

См. [текущий статус сдаваемых этапов](VKR_MILESTONES_1_2.md) и [методику](VKR_STAGE2.md).
