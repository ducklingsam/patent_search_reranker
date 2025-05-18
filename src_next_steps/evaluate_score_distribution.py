import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.reranker import PatentReranker

def parse_args():
    parser = argparse.ArgumentParser(
        description="Сбор распределения скореров ранкера для релевантных и нерелевантных документов"
    )
    parser.add_argument(
        "--manual_csv", required=True,
        help="Путь к manual_qrels.csv с колонками query, id, label"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Путь к файлу модели ранкера LightGBM"
    )
    parser.add_argument(
        "--top_k", type=int, default=100,
        help="Сколько документов запрашивать у RosPatent (по умолчанию 100)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Загрузка Qrels
    qrels = pd.read_csv(args.manual_csv)

    # Инициализация ранкера
    reranker = PatentReranker(args.model_path)

    all_scores = []
    all_labels = []

    for query in qrels['query'].unique():
        # Qrels для данного запроса
        df_q = qrels[qrels['query'] == query][['id', 'label']]
        # Предсказания ранкера
        df_feats, _ = reranker.predict(query, top_k=args.top_k)

        # Сливаем метки (отсутсвующие считаем 0)
        df_merged = df_feats.merge(df_q, on='id', how='left').fillna({'label': 0})
        # Собираем списки
        all_scores.extend(df_merged['score'].tolist())
        all_labels.extend(df_merged['label'].tolist())

    # Датафрейм для удобства
    df_scores = pd.DataFrame({
        'score': all_scores,
        'label': all_labels
    })

    # Рисуем гистограмму
    plt.hist(
        [df_scores[df_scores['label'] == 1]['score'],
         df_scores[df_scores['label'] == 0]['score']],
        bins=50,
        label=['релевантные', 'нерелевантные'],
        alpha=0.5
    )
    plt.xlabel('Score ранкера')
    plt.ylabel('Частота')
    plt.title('Распределение скореров для релевантных и нерелевантных документов')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
