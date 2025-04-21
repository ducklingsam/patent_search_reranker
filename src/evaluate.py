import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from .reranker import PatentReranker
from .metrics import precision_at_k, mrr_at_k, recall_at_k, ndcg_at_k


def evaluate_manual(manual_csv: str, reranker_model_path: str):
    """
    Оценка модели PatentReranker с использованием Leave-One-Query-Out.
    Метрики: Precision@5, MRR@20, Recall@20, NDCG@5.
    """
    df = pd.read_csv(manual_csv)
    model = PatentReranker(reranker_model_path)
    logo = LeaveOneGroupOut()
    queries = df['query'].unique()
    groups = df['query']
    results = []

    for train_idx, test_idx in logo.split(df, groups=groups):
        print("Evaluating...")
        test_df = df.iloc[test_idx]
        q = test_df['query'].iloc[0]
        relevant = set(test_df[test_df['label'] == 1]['id'])
        preds = model.predict(q, top_k=20)
        ranked_ids = preds['id'].tolist()

        results.append({
            'query': q,
            'Precision@5': precision_at_k(ranked_ids, relevant, k=5),
            'MRR@20': mrr_at_k(ranked_ids, relevant, k=20),
            'Recall@20': recall_at_k(ranked_ids, relevant, k=20),
            'NDCG@5': ndcg_at_k(ranked_ids, relevant, k=5)
        })

    df_res = pd.DataFrame(results)
    avg_metrics = df_res[['Precision@5', 'MRR@20', 'Recall@20', 'NDCG@5']].mean().to_dict()
    print("Leave-One-Query-Out Evaluation:")
    print(df_res.to_markdown(index=False))
    print("\nAverage Metrics:")
    print(pd.DataFrame([avg_metrics]).to_markdown(index=False))
    df_res.to_csv('evaluation_results.csv', index=False)
    return df_res
