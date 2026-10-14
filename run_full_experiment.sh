#!/bin/bash

echo " ЗАПУСК ПОЛНОГО ЦИКЛА ЭКСПЕРИМЕНТОВ NODE2VEC"
echo "=============================================="

# Проверяем Python
if ! command -v python &> /dev/null; then
    echo " Python не найден. Установите Python 3.8+"
    exit 1
fi

# Проверяем виртуальное окружение
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "  Виртуальное окружение не активировано"
    echo "Запустите: source .venv/bin/activate"
    exit 1
fi

echo " Виртуальное окружение активно: $VIRTUAL_ENV"

# Этап 1: Диагностика
echo ""
echo " ЭТАП 1: Диагностика системы"
python -m src.debug_check
if [ $? -ne 0 ]; then
    echo " Ошибка диагностики. Проверьте файлы данных."
    exit 1
fi

# Этап 2: Конвертация данных
echo ""
echo " ЭТАП 2: Конвертация данных Book-Crossing"
python -m src.convert
if [ $? -ne 0 ]; then
    echo " Ошибка конвертации. Проверьте BX-Books.csv и BX-Book-Ratings.csv"
    exit 1
fi

# Этап 3: Создание графа
echo ""
echo "🕸  ЭТАП 3: Создание и сохранение графа"
python -m src.data_loader
if [ $? -ne 0 ]; then
    echo " Ошибка создания графа"
    exit 1
fi

# Этап 4: Эксперименты Node2Vec
echo ""
echo " ЭТАП 4: Проведение экспериментов Node2Vec"
python -m src.experiments_updated
if [ $? -ne 0 ]; then
    echo " Ошибка экспериментов"
    exit 1
fi

# Этап 5: Визуализации
echo ""
echo " ЭТАП 5: Создание визуализаций"
python -m src.create_visualizations
if [ $? -ne 0 ]; then
    echo " Ошибка создания визуализаций"
    exit 1
fi

# Этап 6: Анализ результатов
echo ""
echo " ЭТАП 6: Анализ результатов и создание отчёта"
python -m src.analyze_results
if [ $? -ne 0 ]; then
    echo " Ошибка анализа результатов"
    exit 1
fi

echo ""
echo " ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!"
echo "================================"
echo "Результаты доступны в папках:"
echo "   results/plots/ - графики и визуализации"
echo "   results/metrics/ - метрики и отчёты"
echo "   results/embeddings/ - векторные представления"
echo ""