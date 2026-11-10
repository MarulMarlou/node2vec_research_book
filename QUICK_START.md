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

## Исследование выбора объяснения и коэффициента α

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations -SkipInstall
```

Файлы `results/explanation_study/explanation_validation_grid.csv` и `explanation_study_report.json` формируются только после **реального** локального запуска. Выбор метода и `alpha` идёт по validation, test не используется для подбора. [Методика](docs/VKR_NEXT_STAGE_2026-11-10.md).

## После анализа candidate ceiling: один split, пулы 25 и 100

**Подтверждено по split=2092:** релевантные test-книги присутствуют в каталоге для 120/120 пользователей, но в исходный top-25 входят только для **39/120**, а в top-100 — для **77/120**; 81/120 пользователей принципиально не могут получить релевантный top10 из пула 25. На test validation-selected informative/rerank α=0,5 NDCG@10=0,033244 против 0,035940 у posthoc; bootstrap CI включает ноль. [Проверенные результаты](docs/VKR_STAGE3_CANDIDATE_CEILING_2026-11-10.md).

**Новый воспроизводимый эксперимент:** один общий train-only граф/Node2Vec, один **новый split=2102**, одинаковые тестовые пользователи, сравнение pool25 и pool100. Для чистого эффекта пула сравниваются оба размера при **одинаковых** методе/α (настроенных на validation для pool25). Отдельно сравниваются две validation-настроенные системы. [Полный протокол](docs/VKR_CANDIDATE_POOL_25_VS_100_PROTOCOL_2026-11-10.md).

\`\`\`powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations-pool-compare -SkipInstall
\`\`\`

Результаты \`results/candidate_pool_study/\` пока **не получены**. Для их анализа нужны JSON и три CSV из указанной папки.

## После парного bootstrap: диагностика ограничения выбора кандидатов

**Реальный split=2082, 80 test-пользователей:** rerank α=0,5 NDCG@10=0,032821 против 0,028946 у posthoc; при этом 95%-й парный bootstrap-CI разницы **[−0,001410; +0,009562]** содержит нулевой эффект. Для 69 из 80 NDCG@10 не изменился, у 66 из 80 оба метода не дали релевантного попадания в top-10. [Разбор и воспроизводимость](docs/VKR_STAGE3_PAIRED_BOOTSTRAP_2026-11-10.md).

Для дальнейшего изучения низких абсолютных метрик:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations-ceiling -SkipInstall
```

В новом `split=2092` измеряются потолки восстановления удержанных положительных оценок: весь каталог эмбеддингов, top-25, top-100 базовых cosine-кандидатов и выбранный пул 25. Test-метки используются **только для оценки**, не для подбора кандидатов или параметров. Результаты локально в `results/explanation_study_ceiling/`, пока **не получены**.

## Новый подтверждённый оптимизированный пилот и парные интервалы

Реальный \`split=2072\`: **0% усечений informative на validation** после структурной оптимизации; лучший \`shortest+rerank α=0,2\`, test NDCG@10=0,066130 против 0,062838 у фиксированного posthoc. [Полный исследовательский отчёт](docs/VKR_STAGE3_OPTIMIZED_RESULTS_2026-11-10.md).

Следующая независимая серия дополнительно сохраняет **пользовательские парные разницы без исходных User-ID** и условные bootstrap-интервалы:

\`\`\`powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations-paired -SkipInstall
\`\`\`

Новый split=2082; до 80 пользователей, результаты в \`results/explanation_study_paired/\`. **Это ещё не выполненный опыт**. Изменение в версии кода нельзя ретроспективно приписывать старому \`split=2072\`.

## После третьей серии: проверка оптимизированного поиска

Получены результаты `explanations-budget` и `faithfulness-extended`. У informative 100% усечений даже при 200/5000; четыре пары с 20 degree-aware контролями каждая дают неодинаковые по силе эффекты и **не являются достаточным статистическим доказательством**. [Подробный отчёт по третьей серии](docs/VKR_STAGE3_SEARCH_BUDGET_AND_EXTENDED_FAITHFULNESS_2026-11-10.md).

Для исправления ограничений DFS добавлен структурный алгоритм поиска четырёхрёберных путей. Проверяйте его **на новом split=2072**, не переиспользуя просмотренные test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations-optimized -SkipInstall
```

Новый результат создаётся в `results/explanation_study_optimized/` и не переписывает предшествующие серии. Это **ещё не выполненный на Book-Crossing запуск**.

## После второго исследования: более строгие проверки

**Новый подтверждённый depth-4 пилот:** на 40 тестовых пользователях `shortest + rerank α=0,5` показал NDCG@10=0,077620, фиксированный posthoc — 0,067643. Покрытие объяснениями: 100% против 94,25%. В informative-поиске **100% усечённых поисков** — это методологический недостаток, а не доказательство превосходства shortest. Faithfulness: обе абляции дали снижение cosine, но одна контрольная абляция второй пары дала большее падение. [Точный анализ](docs/VKR_STAGE3_DEPTH4_FAITHFULNESS_2026-11-10.md).

```powershell
# Новый split seed=2062, max_depth=4, max_paths=200, max_expansions=5000
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations-budget -SkipInstall

# До 4 пар и 20 контрольных удалений для каждой; новая отдельная папка
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode faithfulness-extended -SkipInstall
```

Результаты: `results/explanation_study_budget/` и `results/faithfulness_extended/`. **Новые исследования ещё нужно запустить на компьютере**; команды требуют установленного окружения, а faithfulness — наличия актуальных `quick/main` артефактов. Новые серии не перезаписывают прежние папки.

## Проверенные пилотные результаты и новый depth-4 эксперимент

На реальных данных 10.11.2026 проведено 8 проверок настроек объяснений на validation, выбран `shortest + posthoc`; итоговый test NDCG@10=0,033914 и покрытие объяснениями 87,5% для 20 пользователей. Отдельный faithfulness-пилот выполнен для двух пар (не является статистическим доказательством). [Сводные результаты и ограничения](docs/VKR_STAGE3_PILOT_2026-11-10.md).

```powershell
# Новый независимый split=2052, пути до 4 рёбер, 40 пользователей
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode explanations-depth4 -SkipInstall

# Новые relation/degree-matched контроли (перезапишут локальные faithfulness файлы)
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode faithfulness -SkipInstall
```

Старые результаты по split=2042 не заменяются новым depth4; обе серии нужно хранить и сравнивать с указанием разных разбиений.

## Новые исследовательские режимы

```powershell
# Демо текущих эмбеддингов с 2D PCA — в пятой вкладке Streamlit
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode app -SkipInstall

# Контролируемое небольшое обогащение ISBN из Open Library (нужен ваш email)
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode genre -SkipInstall -ContactEmail "your-email@example.org" -GenreBatchSize 12

# Явно перезаписать quick-модель с новым genre CSV
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode quick -SkipInstall -UseGenres

# Эксперимент удаления подтверждённых рёбер и повторного обучения
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode faithfulness -SkipInstall
```

**Использование:** новые экспериментальные результаты сохраняются только локально. Массовые запросы к Open Library API недопустимы; жанры требуют оценки покрытия и проверки источников. **[Методика новых функций](docs/VKR_NEXT_STAGE_2026-11-10.md).**

## 7. Сравнение Node2Vec и TransE (новый второй этап)

```powershell
# 4 пробных сравнения: 2 Node2Vec и 2 TransE
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode compare-quick -SkipInstall

# 42 подготовленных сравнительных эксперимента
powershell -ExecutionPolicy Bypass -File .\scripts\run_vkr_windows.ps1 -Mode compare-full -SkipInstall
```

Результаты сохраняются в `results/comparison/comparison_report.json` и `comparison_experiments.csv`. Сравниваются **одинаковые** train/validation/test и каталоги; ROC-AUC рассчитан на явно оценённых высоко/низко книгах, не на отсутствующих оценках. **Наличие 42 конфигураций не означает, что фактические запуски уже проведены.** См. [аудит исходных требований](docs/VKR_REQUIREMENTS_AUDIT_2026-10-28.md).
