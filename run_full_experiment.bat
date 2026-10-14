@echo off
REM

echo  ЗАПУСК ПОЛНОГО ЦИКЛА ЭКСПЕРИМЕНТОВ NODE2VEC
echo ==============================================

REM Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python не найден. Установите Python 3.8+
    pause
    exit /b 1
)

REM Проверяем виртуальное окружение
if "%VIRTUAL_ENV%"=="" (
    echo   Виртуальное окружение не активировано
    echo Запустите: .venv\Scripts\activate
    pause
    exit /b 1
)

echo  Виртуальное окружение активно: %VIRTUAL_ENV%

REM Этап 1: Диагностика
echo.
echo  ЭТАП 1: Диагностика системы
python -m src.debug_check
if errorlevel 1 (
    echo  Ошибка диагностики. Проверьте файлы данных.
    pause
    exit /b 1
)

REM Этап 2: Конвертация данных
echo.
echo  ЭТАП 2: Конвертация данных Book-Crossing
python -m src.convert
if errorlevel 1 (
    echo  Ошибка конвертации. Проверьте BX-Books.csv и BX-Book-Ratings.csv
    pause
    exit /b 1
)

REM Этап 3: Создание графа
echo.
echo ️  ЭТАП 3: Создание и сохранение графа
python -m src.data_loader
if errorlevel 1 (
    echo  Ошибка создания графа
    pause
    exit /b 1
)

REM Этап 4: Эксперименты Node2Vec
echo.
echo  ЭТАП 4: Проведение экспериментов Node2Vec
python -m src.experiments_updated
if errorlevel 1 (
    echo  Ошибка экспериментов
    pause
    exit /b 1
)

REM Этап 5: Визуализации
echo.
echo  ЭТАП 5: Создание визуализаций
python -m src.create_visualizations
if errorlevel 1 (
    echo  Ошибка создания визуализаций
    pause
    exit /b 1
)

REM Этап 6: Анализ результатов
echo.
echo  ЭТАП 6: Анализ результатов и создание отчёта
python -m src.analyze_results
if errorlevel 1 (
    echo  Ошибка анализа результатов
    pause
    exit /b 1
)

echo.
echo  ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!
echo ================================
echo Результаты доступны в папках:
echo    results/plots/ - графики и визуализации
echo    results/metrics/ - метрики и отчёты
echo    results/embeddings/ - векторные представления
echo    node2vec_research_report.pdf - готовый отчёт
echo.
pause