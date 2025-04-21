# Patent Search & Re-ranker

## Описание
Система поиска и ранжирования патентов Роспатента с дообученными эмбеддингами,
использующим LightGBM LambdaRank и мультиметрическими признаками:
- BM25 (from API)
- Dense similarity (дообученная RuBERT-модель)
- IPC similarity (автоматически inferred)

## Установка
```bash
git clone <repo_url>
cd patent_search_re_reranker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# заполнить в .env ключи
```

## Использование
### 1. Дообучение эмбеддингов
```bash
python -m src.main embed --gold data/gold_labels.csv --output models/contrastive-rubert --epochs 3
```

### 2. Обучение ранкера
```bash
python -m src.main train --gold data/gold_labels.csv --output models/patent_reranker.txt
```

### 3. Оценка и абляции
```bash
# manual_qrels.csv должен содержать ручную разметку для честного теста
python -m src.main eval --manual data/manual_qrels.csv --model models/patent_reranker.txt
```

### 4. Эксперименты
```bash
python -m src.main experiment --gold data/gold_labels.csv --manual manual_qrels.csv --model models/patent_reranker.txt
```

## Структура данных
- `data/gold_labels.csv`: LLM-сгенерированные positive/negative для тренировки
- `data/manual_qrels.csv`: вручную размеченная выборка для финального теста

## Результаты
Графики и отчёты сохраняются в notebooks/analysis.ipynb
"""
