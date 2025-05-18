import argparse
import pandas as pd
from src.reranker import PatentReranker
from src.metrics import ndcg_at_k, mrr_at_k


def parse_args():
    parser = argparse.ArgumentParser(
        description="Базовая оценка пайплайна PatentReranker: NDCG@1,5,10 и MRR@20"
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
    parser.add_argument(
        "--output_csv", default="baseline_evaluation.csv",
        help="Файл для сохранения результатов (CSV)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Загрузка Qrels и приведение id к строке
    df = pd.read_csv(args.manual_csv)
    df['id'] = df['id'].astype(str)
    # Инициализация модели
    reranker = PatentReranker(args.model_path)

    results = []
    for query in df['query'].unique():
        # Собираем релевантные id как строки
        relevant = set(df[(df['query'] == query) & (df['label'] == 1)]['id'])
        if not relevant:
            print(f"Пропускаем запрос '{query}', нет положительных меток.")
            continue
        # Предсказания ранкера
        df_feats, patents = reranker.predict(query, top_k=args.top_k)
        # Приводим id из фичей к строке
        df_feats['id'] = df_feats['id'].astype(str)
        ranked_ids = df_feats['id'].tolist()
        print("Релевантные id:", relevant)
        print("Топ-10 id от ранкера:", ranked_ids)
        print("Пересечение:", relevant.intersection(ranked_ids))
        try:
            ndcg1 = ndcg_at_k(ranked_ids, relevant, k=1)
            ndcg5 = ndcg_at_k(ranked_ids, relevant, k=5)
            ndcg10 = ndcg_at_k(ranked_ids, relevant, k=10)
            mrr20 = mrr_at_k(ranked_ids, relevant, k=20)
            results.append({
                'query': query,
                'NDCG@1': ndcg1,
                'NDCG@5': ndcg5,
                'NDCG@10': ndcg10,
                'MRR@20': mrr20
            })
        except Exception as e:
            pass

    # Сохраняем результаты
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output_csv, index=False)

    # Выводим в консоль
    if not df_res.empty:
        print("\nРезультаты по запросам:")
        print(df_res.to_markdown(index=False))
        print("\nСредние метрики:")
        print(df_res[['NDCG@1', 'NDCG@5', 'NDCG@10', 'MRR@20']].mean().to_markdown())
    else:
        print("Нет данных для оценки.")


if __name__ == '__main__':
    main()
