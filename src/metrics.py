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