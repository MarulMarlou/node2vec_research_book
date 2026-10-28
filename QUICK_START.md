# Быстрый запуск актуальной ВКР (Windows)

**Это руководство относится к ветке** `feature/vkr-stage1-typed-graph-holdout`. Для полной информации см. [README](README.md) и [пошаговую инструкцию Windows](docs/VKR_WINDOWS.md).

## 1. Установка

- Python **3.11** (не 3.15), Git, Windows PowerShell.
- Исходные CSV Book-Crossing: `BX-Books.csv`, `BX-Book-Ratings.csv` (папка `data/raw/`). [Источник данных и описание — GroupLens](https://grouplens.org/datasets/book-crossing/). Файлы не входят в репозиторий.

Если Python 3.11 ещё отсутствует:

```powershell
winget install --id Python.Python.3.11 --exact --source winget --accept-package-agreements --accept-source-agreements
```

Перезапустите PowerShell, проверьте `py -3.11 --version`.

## 2. Скачать **эту** ветку

```powershell
git clone --branch feature/vkr-stage1-typed-graph-holdout --single-branch https://github.com/MarulMarlou/node2vec_research_book.git
cd node2vec_research_book
```

При скачивании ZIP вместо clone откройте [ветку ВКР](https://github.com/MarulMarlou/node2vec_research_book/tree/feature/vkr-stage1-typed-graph-holdout), нажмите **Code → Download ZIP** и распакуйте архив.

## 3. Данные

```text
node2vec_research_book/
  data/
    raw/
      BX-Books.csv
      BX-Book-Ratings.csv
```

Имена файлов должны совпадать **точно**. CSV имеют формат `latin-1`, разделитель `;`. Не открывайте и не пересохраняйте их в Excel.

## 4. Построить граф и обучить Node2Vec

В PowerShell из корня проекта:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode quick
```

При первом запуске автоматически создаётся `.venv-vkr` и устанавливаются зависимости. Режим `quick`: до 150 пользователей и 1 000 книг. После успешного завершения в консоли будет `audit_passed: true`. Ошибка `Python 3.11 required` означает, что версия 3.11 не обнаружена командой `py -3.11`.

## 5. Посмотреть демонстрацию

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app -SkipInstall
```

Откройте http://localhost:8501. Сначала обязательно выполните `quick` или `main` — без графа и эмбеддингов интерфейсу нечего показывать.

## 6. Отдельное исследование эмбеддингов

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode study-quick -SkipInstall
```

Результаты: `results/study/study_report.json` и `results/study/embedding_experiments.csv` — **3 конфигурации Node2Vec**, selection по validation и однократное вычисление test.

Режим `study-full` — **36 запусков Node2Vec на уменьшенном графе**; может выполняться заметно дольше. Он не является сравнением с TransE и **не закрывает второй этап самостоятельно**.

Другие команды: `-Mode main` — до 1 000 пользователей / 5 000 книг; `-Mode tests` — тесты. При повторных запусках `quick`/`main` основные артефакты перезаписываются. Для повторных запусков используйте `-SkipInstall` после установки зависимостей.

## 7. Сравнение Node2Vec и TransE (новый второй этап)

```powershell
# 4 пробных сравнения: 2 Node2Vec и 2 TransE
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode compare-quick -SkipInstall

# 42 подготовленных сравнительных эксперимента
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode compare-full -SkipInstall
```

Результаты сохраняются в `results/comparison/comparison_report.json` и `comparison_experiments.csv`. Сравниваются **одинаковые** train/validation/test и каталоги; ROC-AUC рассчитан на явно оценённых высоко/низко книгах, не на отсутствующих оценках. **Наличие 42 конфигураций не означает, что фактические запуски уже проведены.** См. [аудит исходных требований](docs/VKR_REQUIREMENTS_AUDIT_2026-10-28.md).
