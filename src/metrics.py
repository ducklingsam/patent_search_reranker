from sklearn.metrics import ndcg_score

def precision_at_k(ranked_ids, relevant_set, k=5):
    """Precision@k: доля релевантных в топ-k."""
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for pid in topk if pid in relevant_set) / k

def mrr_at_k(ranked_ids, relevant_set, k=20):
    """MRR@k: средняя обратная ранга первого релевантного документа в топ-k."""
    for idx, pid in enumerate(ranked_ids[:k], start=1):
        if pid in relevant_set:
            return 1.0 / idx
    return 0.0

def recall_at_k(ranked_ids, relevant_set, k=20):
    """Recall@k: доля релевантных документов из топ-k."""
    topk = ranked_ids[:k]
    if not relevant_set:
        return 0.0
    return sum(1 for pid in topk if pid in relevant_set) / len(relevant_set)

def ndcg_at_k(ranked_ids, relevant_set, k=5):
    """NDCG@k: нормализованная дисконтированная кумулятивная выгода."""
    true_relevance = [1 if pid in relevant_set else 0 for pid in ranked_ids[:k]]
    if sum(true_relevance) == 0:
        return 0.0
    return ndcg_score([true_relevance], [true_relevance], k=k)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sentence_transformers.cross_encoder import CrossEncoder
    from sklearn.metrics import ndcg_score

    # 1. Загрузка данных
    qrels = pd.read_csv('../data/labeled_patents.csv')  # topic, query, patent_id, label
    cands = pd.read_csv('../data/query_candidates_top50.csv')  # topic, query, id, bi_score, label
    corpus = pd.read_csv('../data/all_patents.csv')[['id', 'text']]
    corpus_map = dict(zip(corpus['id'], corpus['text']))

    # 2. Инициализация классификатора
    ce = CrossEncoder('../models/crossencoder_patent_fast', device='mps')

    # 3. Для каждой темы считаем би-ранж и soft-prob CE-ранж
    res = []
    for topic, grp in cands.groupby('topic'):
        qs = [grp['query'].iloc[0]] * len(grp)
        docs = [corpus_map[i] for i in grp['id']]
        # получаем вероятности второго (положительного) класса
        probs = ce.predict(list(zip(qs, docs)), apply_softmax=True)[:, 1]

        grp = grp.copy()
        grp['ce_prob'] = probs

        # ground-truth порядок
        y_true = grp.sort_values('bi_score', ascending=False)['label'].values
        # два ранжирования
        y_bi = y_true
        y_ce = grp.sort_values('ce_prob', ascending=False)['label'].values

        res.append({
            'ndcg_bi': ndcg_score([y_true], [y_bi], k=10),
            'ndcg_ce': ndcg_score([y_true], [y_ce], k=10),
            'mrr_bi': 1 / (np.where(y_bi == 1)[0][0] + 1) if 1 in y_bi else 0,
            'mrr_ce': 1 / (np.where(y_ce == 1)[0][0] + 1) if 1 in y_ce else 0
        })

    df = pd.DataFrame(res)
    print("nDCG@10 — bi:", df['ndcg_bi'].mean(),
          "  bi+CE_prob:", df['ndcg_ce'].mean())
    print("MRR    — bi:", df['mrr_bi'].mean(),
          "  bi+CE_prob:", df['mrr_ce'].mean())
