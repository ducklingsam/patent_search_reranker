#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from peft import get_peft_model, LoraConfig, TaskType
from src.rosclient import RosPatentClient
from src.utils import device

def prepare_model(model_name="DeepPavlov/rubert-base-cased"):
    model = SentenceTransformer(model_name)
    transformer = model[0]
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=4, lora_alpha=16, lora_dropout=0.05
    )
    transformer.auto_model = get_peft_model(transformer.auto_model, lora_cfg)
    model[0] = transformer
    return model

def extract_text_from_doc(doc: dict) -> str:
    """
    Берёт из ответа API текст абстракта:
    - если doc == {"ru": ..., "en": ...}, вернёт doc["ru"] или, если пусто, doc["en"];
    - если doc содержит ключ "abstract", то аналогично в doc["abstract"];
    - иначе конкатенирует все строковые значения в doc.
    """
    # случай, когда get_document() сразу вернул {'ru':..., 'en':...}
    if set(doc.keys()) >= {"ru", "en"}:
        return doc.get("ru", "") or doc.get("en", "") or ""
    # случай, когда в корне есть "abstract"
    if "abstract" in doc:
        abstr = doc["abstract"]
        if isinstance(abstr, dict):
            return abstr.get("ru", "") or abstr.get("en", "") or ""
        elif isinstance(abstr, str):
            return abstr
    # fallback — склеим все текстовые поля
    parts = [v for v in doc.values() if isinstance(v, str)]
    return " ".join(parts)

def fetch_texts_from_api(client: RosPatentClient, ids: list[str]) -> dict[str, str]:
    texts = {}
    for pid in set(ids):
        doc = client.get_document(str(pid))
        txt = extract_text_from_doc(doc)
        if not txt:
            raise RuntimeError(f"Не удалось получить текст для патента {pid}")
        texts[pid] = txt
    return texts

def load_positive_pairs(df: pd.DataFrame, client: RosPatentClient) -> list[InputExample]:
    all_ids = df['id'].astype(str).tolist()
    texts = fetch_texts_from_api(client, all_ids)
    examples = []
    for _, row in df[df.label == 1].iterrows():
        pid = str(row.id)
        examples.append(InputExample(texts=[row.query, texts[pid]]))
    return examples

def build_ir_evaluator(df_dev: pd.DataFrame, client: RosPatentClient) -> InformationRetrievalEvaluator:
    all_ids = df_dev['id'].astype(str).tolist()
    texts = fetch_texts_from_api(client, all_ids)
    uniq_q = df_dev['query'].unique().tolist()
    queries = { f"q{i}": q for i, q in enumerate(uniq_q) }
    corpus  = { str(rid): texts[str(rid)] for rid in all_ids }
    qrels   = {}
    for q, grp in df_dev.groupby("query"):
        qid = f"q{uniq_q.index(q)}"
        qrels[qid] = { str(rid): int(lbl) for rid,lbl in zip(grp['id'], grp['label']) }
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name="dev-ir"
    )

def mine_hard_triplets(model, df_train: pd.DataFrame, client: RosPatentClient, top_k: int) -> list[InputExample]:
    all_ids   = df_train['id'].astype(str).tolist()
    texts     = fetch_texts_from_api(client, all_ids)
    queries   = df_train['query'].tolist()
    labels    = df_train['label'].tolist()
    doc_texts = [texts[pid] for pid in all_ids]

    corpus_emb = model.encode(doc_texts, convert_to_tensor=True)
    query_emb  = model.encode(queries,   convert_to_tensor=True)
    sims       = torch.matmul(query_emb, corpus_emb.T)
    _, topi    = torch.topk(sims, k=top_k+1, dim=1)

    triplets = []
    for i, q in enumerate(queries):
        pos_idxs = [j for j,(qq,lbl) in enumerate(zip(queries, labels)) if qq==q and lbl==1]
        if not pos_idxs:
            continue
        pos_text = doc_texts[pos_idxs[0]]
        cnt = 0
        for cand in topi[i].cpu().tolist():
            if cand in pos_idxs:
                continue
            triplets.append(InputExample(texts=[q, pos_text, doc_texts[cand]]))
            cnt += 1
            if cnt >= top_k:
                break
    return triplets

def patch_triplet_forward():
    from sentence_transformers.losses import TripletLoss
    def patched(self, sentence_features, labels=None):
        reps = [ self.model(f)["sentence_embedding"] for f in sentence_features ]
        a = torch.stack(reps[0::3]); p = torch.stack(reps[1::3]); n = torch.stack(reps[2::3])
        dpos = self.distance_metric(a, p); dneg = self.distance_metric(a, n)
        return F.relu(dpos - dneg + self.triplet_margin).mean()
    TripletLoss.forward = patched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_qrels", required=True, help="CSV с колонками [query,id,label]")
    parser.add_argument("--output_dir",  default="models/tiny_rubert_lora")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--epochs1",     type=int, default=10)
    parser.add_argument("--warmup1",     type=int, default=100)
    parser.add_argument("--epochs2",     type=int, default=5)
    parser.add_argument("--warmup2",     type=int, default=50)
    parser.add_argument("--top_k",       type=int, default=10)
    parser.add_argument("--margin",      type=float, default=1.0)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # сплитим train/dev 80/20
    df_all   = pd.read_csv(args.train_qrels)
    df_dev   = df_all.sample(frac=0.2, random_state=42).reset_index(drop=True)
    df_train = df_all.drop(df_dev.index).reset_index(drop=True)

    client = RosPatentClient()
    model  = prepare_model()
    model.to(device)

    # === СТАДИЯ 1: MNRLoss + IR-evaluator ===
    train_examples = load_positive_pairs(df_train, client)
    print(f"Stage1: {len(train_examples)} positive pairs")
    loader1 = DataLoader(train_examples, batch_size=args.batch_size,
                         shuffle=True, collate_fn=model.smart_batching_collate)
    loss1     = losses.MultipleNegativesRankingLoss(model)
    evaluator = build_ir_evaluator(df_dev, client)

    model.fit(
        train_objectives=[(loader1, loss1)],
        epochs=args.epochs1, warmup_steps=args.warmup1,
        evaluator=evaluator, evaluation_steps=50,
        output_path=os.path.join(args.output_dir, "stage1"),
        show_progress_bar=True
    )

    # === СТАДИЯ 2: hard negatives + TripletLoss ===
    print("Mining hard negatives…")
    triplets = mine_hard_triplets(model, df_train, client, args.top_k)
    print(f"Stage2: {len(triplets)} triplets")
    patch_triplet_forward()
    loader2 = DataLoader(triplets, batch_size=args.batch_size,
                         shuffle=True, collate_fn=model.smart_batching_collate)
    loss2 = losses.TripletLoss(model=model, triplet_margin=args.margin)
    model.to(device)
    model.fit(
        train_objectives=[(loader2, loss2)],
        epochs=args.epochs2, warmup_steps=args.warmup2,
        evaluator=evaluator, evaluation_steps=50,
        output_path=os.path.join(args.output_dir, "stage2"),
        show_progress_bar=True
    )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
