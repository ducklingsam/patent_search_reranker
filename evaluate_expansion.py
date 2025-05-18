import argparse
import asyncio
import pandas as pd
from src.reranker import PatentReranker
from src.rosclient import RosPatentClient
from src.features import extract_features
from src.metrics import ndcg_at_k
from usage_example import preprocess_query, expand_query

TOP_K_SEARCH = 100
TOP_K_RANK = 100


def recall_at_k(ranked_ids, relevant, k=100):
    hits = sum(1 for pid in ranked_ids[:k] if pid in relevant)
    return hits / len(relevant) if relevant else 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Оценка пайплайна с GPT-preprocessing + expansion (recall@100, NDCG@5)"
    )
    p.add_argument("--manual_csv", required=True, help="manual_qrels.csv")
    p.add_argument("--model_path", required=True, help="LGBM модель ранкера")
    p.add_argument("--output_csv", default="expansion_eval.csv", help="Файл с детальными результатами")
    return p.parse_args()


async def process_query(query, reranker, ros_client):
    # 1. preprocessing & expansion
    preprocessed = await preprocess_query(query)
    expansions = await expand_query(preprocessed)
    phrases = [preprocessed] + expansions
    combined_q = " OR ".join([f"({p})" for p in phrases])

    # 2. search
    raw = ros_client.search_raw({"q": combined_q, "limit": TOP_K_SEARCH, "offset": 0})
    patents = raw.get("hits", [])

    # 3. features & ranking
    feats = extract_features(preprocessed, patents)
    df_feats = pd.DataFrame(feats)
    if df_feats.empty:
        return [], []  # no candidates
    features = [c for c in df_feats.columns if c not in ("id", "year")]
    df_feats["score"] = reranker.model.predict(df_feats[features])
    df_feats = df_feats.sort_values("score", ascending=False)
    ranked_ids = df_feats["id"].tolist()
    return ranked_ids, patents


def main():
    args = parse_args()
    qrels = pd.read_csv(args.manual_csv)
    reranker = PatentReranker(args.model_path)
    ros_client = RosPatentClient()

    loop = asyncio.get_event_loop()

    results = []
    for query in qrels["query"].unique():
        relevant = set(qrels[(qrels["query"] == query) & (qrels["label"] == 1)]["id"])
        if not relevant:
            continue
        ranked_ids, _ = loop.run_until_complete(process_query(query, reranker, ros_client))
        if not ranked_ids:
            rec100 = 0
            ndcg5 = 0
        else:
            rec100 = recall_at_k(ranked_ids, relevant, k=TOP_K_RANK)
            ndcg5 = ndcg_at_k(ranked_ids, relevant, k=5)
        results.append({"query": query, "Recall@100": rec100, "NDCG@5": ndcg5})

    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output_csv, index=False)

    if not df_res.empty:
        print("\nСредние метрики:")
        print(df_res[["Recall@100", "NDCG@5"]].mean().to_markdown())
    else:
        print("Нет данных для оценки.")


if __name__ == "__main__":
    main()
