import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
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
        "--top_k", type=int, default=20,
        help="Сколько документов запрашивать у RosPatent (по умолчанию 20)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    qrels = pd.read_csv(args.manual_csv)

    reranker = PatentReranker(args.model_path)

    all_scores = []
    all_labels = []

    for query in qrels['query'].unique():
        df_q = qrels[qrels['query'] == query][['id', 'label']]
        df_feats, _ = reranker.predict(query, top_k=args.top_k)

        df_merged = df_feats.merge(df_q, on='id', how='left').fillna({'label': 0})
        all_scores.extend(df_merged['score'].tolist())
        all_labels.extend(df_merged['label'].tolist())

    df_scores = pd.DataFrame({
        'score': all_scores,
        'label': all_labels
    })

    best_thresh = 0.0
    best_f1 = 0.0

    for t in [x / 100 for x in range(0, 101)]:
        preds = [1 if s >= t else 0 for s in all_scores]
        _, _, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Лучший threshold: {best_thresh:.2f} с F1: {best_f1:.4f}") # Result: Лучший threshold: 0.66 с F1: 0.6667

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
    plt.savefig('results_new/threshold_f1.png', dpi=300)

if __name__ == "__main__":
    main()
