import pandas as pd
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor, as_completed
from .reranker import PatentReranker
from .metrics import precision_at_k, mrr_at_k, recall_at_k, ndcg_at_k
from .features import extract_features
from .rosclient import RosPatentClient


def baseline_bm25(query, patents, relevant_set, k=20):
    df = pd.DataFrame(patents)
    df['score'] = df['bm25']
    ranked = df.sort_values('score', ascending=False)
    ranked_ids = ranked['id'].tolist()
    return {
        'Precision@5': precision_at_k(ranked_ids, relevant_set, k=5),
        'MRR@20': mrr_at_k(ranked_ids, relevant_set, k=20),
        'Recall@20': recall_at_k(ranked_ids, relevant_set, k=20),
        'NDCG@5': ndcg_at_k(ranked_ids, relevant_set, k=5)
    }


def baseline_dense(query, patents, relevant_set, k=20):
    df = pd.DataFrame(patents)
    df['score'] = df['dense_sim']
    ranked = df.sort_values('score', ascending=False)
    ranked_ids = ranked['id'].tolist()
    return {
        'Precision@5': precision_at_k(ranked_ids, relevant_set, k=5),
        'MRR@20': mrr_at_k(ranked_ids, relevant_set, k=20),
        'Recall@20': recall_at_k(ranked_ids, relevant_set, k=20),
        'NDCG@5': ndcg_at_k(ranked_ids, relevant_set, k=5)
    }


def fetch_features_for_query(client, df, q):
    sub = df[df['query'] == q]
    patents = client.search(q, limit=len(sub))
    feats = extract_features(q, patents)
    rows = []
    for f in feats:
        pid = f['id']
        label_series = sub[sub['id'] == pid]['label']
        label = int(label_series.iloc[0]) if not label_series.empty else 0
        rows.append({**f, 'label': label, 'query': q})
    return rows, len(patents)


def ablation_study(gold_csv, output_path, feature_sets=None):
    if feature_sets is None:
        feature_sets = [
            ['bm25', 'dense_sim', 'ipc_sim'],
            ['bm25', 'dense_sim'],
            ['bm25', 'ipc_sim'],
            ['dense_sim', 'ipc_sim']
        ]

    df = pd.read_csv(gold_csv)
    client = RosPatentClient()
    results = []

    for features in feature_sets:
        rows = []
        groups = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_features_for_query, client, df, q): q for q in df['query'].unique()}
            for future in as_completed(futures):
                query = futures[future]
                query_rows, group_size = future.result()
                rows.extend(query_rows)
                groups.append(group_size)

        data = pd.DataFrame(rows)
        X = data[features]
        y = data['label']
        train_data = lgb.Dataset(X, y, group=groups)
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 1
        }
        model = lgb.train(params, train_data, num_boost_round=100)

        reranker = PatentReranker()
        reranker.model = model
        metrics = []

        for q in df['query'].unique():
            sub = df[df['query'] == q]
            relevant = set(sub[sub['label'] == 1]['id'])
            preds, _ = reranker.predict(q, top_k=20, features=features)
            ranked_ids = preds['id'].tolist()
            metrics.append({
                'query': q,
                'Precision@5': precision_at_k(ranked_ids, relevant, k=5),
                'MRR@20': mrr_at_k(ranked_ids, relevant, k=20),
                'Recall@20': recall_at_k(ranked_ids, relevant, k=20),
                'NDCG@5': ndcg_at_k(ranked_ids, relevant, k=5)
            })

        metrics_df = pd.DataFrame(metrics)
        results.append({
            'Features': features,
            'Avg Precision@5': metrics_df['Precision@5'].mean(),
            'Avg MRR@20': metrics_df['MRR@20'].mean(),
            'Avg Recall@20': metrics_df['Recall@20'].mean(),
            'Avg NDCG@5': metrics_df['NDCG@5'].mean()
        })

    results_df = pd.DataFrame(results)
    print("Ablation Study Results:")
    print(results_df.to_markdown(index=False))
    results_df.to_csv(output_path, index=False)
    return results_df


def hyperparameter_tuning(gold_csv, output_path, param_grid=None):
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63]
        }

    df = pd.read_csv(gold_csv)
    client = RosPatentClient()
    results = []

    for lr in param_grid['learning_rate']:
        for leaves in param_grid['num_leaves']:
            rows = []
            groups = []

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(fetch_features_for_query, client, df, q): q for q in df['query'].unique()}
                for future in as_completed(futures):
                    query_rows, group_size = future.result()
                    rows.extend(query_rows)
                    groups.append(group_size)

            data = pd.DataFrame(rows)
            X = data[['bm25', 'dense_sim', 'ipc_sim']]
            y = data['label']
            train_data = lgb.Dataset(X, y, group=groups)
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [5],
                'learning_rate': lr,
                'num_leaves': leaves,
                'min_data_in_leaf': 1
            }
            model = lgb.train(params, train_data, num_boost_round=100)

            reranker = PatentReranker()
            reranker.model = model
            metrics = []

            for q in df['query'].unique():
                sub = df[df['query'] == q]
                relevant = set(sub[sub['label'] == 1]['id'])
                preds, _ = reranker.predict(q, top_k=20)
                ranked_ids = preds['id'].tolist()
                metrics.append({
                    'query': q,
                    'Precision@5': precision_at_k(ranked_ids, relevant, k=5),
                    'MRR@20': mrr_at_k(ranked_ids, relevant, k=20),
                    'Recall@20': recall_at_k(ranked_ids, relevant, k=20),
                    'NDCG@5': ndcg_at_k(ranked_ids, relevant, k=5)
                })

            metrics_df = pd.DataFrame(metrics)
            results.append({
                'Learning Rate': lr,
                'Num Leaves': leaves,
                'Avg Precision@5': metrics_df['Precision@5'].mean(),
                'Avg MRR@20': metrics_df['MRR@20'].mean(),
                'Avg Recall@20': metrics_df['Recall@20'].mean(),
                'Avg NDCG@5': metrics_df['NDCG@5'].mean()
            })

    results_df = pd.DataFrame(results)
    print("Hyperparameter Tuning Results:")
    print(results_df.to_markdown(index=False))
    results_df.to_csv(output_path, index=False)
    return results_df


def compare_methods(manual_csv, reranker_model_path):
    df = pd.read_csv(manual_csv)
    client = RosPatentClient()
    reranker = PatentReranker(reranker_model_path)
    results = []

    def evaluate_query(q):
        sub = df[df['query'] == q]
        relevant = set(sub[sub['label'] == 1]['id'])
        patents = client.search(q, limit=20)
        feats = extract_features(q, patents)
        preds, _ = reranker.predict(q, top_k=20)
        ranked_ids = preds['id'].tolist()
        return [
            {
                'Method': 'Proposed',
                'Query': q,
                'Precision@5': precision_at_k(ranked_ids, relevant, k=5),
                'MRR@20': mrr_at_k(ranked_ids, relevant, k=20),
                'Recall@20': recall_at_k(ranked_ids, relevant, k=20),
                'NDCG@5': ndcg_at_k(ranked_ids, relevant, k=5)
            },
            {
                **baseline_bm25(q, feats, relevant),
                'Method': 'BM25',
                'Query': q
            },
            {
                **baseline_dense(q, feats, relevant),
                'Method': 'Dense',
                'Query': q
            }
        ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(evaluate_query, q) for q in df['query'].unique()]
        for f in as_completed(futures):
            results.extend(f.result())

    results_df = pd.DataFrame(results)
    avg_metrics = results_df.groupby('Method')[['Precision@5', 'MRR@20', 'Recall@20', 'NDCG@5']].mean().reset_index()
    print("Comparison of Methods:")
    print(results_df.to_markdown(index=False))
    print("\nAverage Metrics by Method:")
    print(avg_metrics.to_markdown(index=False))
    results_df.to_csv('comparison_results_new.csv', index=False)
    return results_df