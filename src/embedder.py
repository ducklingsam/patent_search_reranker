import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from rank_bm25 import BM25Okapi
from typing import List

from .rosclient import RosPatentClient
from .utils import device

device = 'cpu'


def train_embedder(gold_csv: str, base_model: str = 'ai-forever/sbert_large_nlu_ru', output_path: str = 'models/contrastive-rubert', epochs: int = 1, batch_size: int = 8):
    df = pd.read_csv(gold_csv)
    client = RosPatentClient()
    examples = []
    for _, row in df[df.label==1].iterrows():
        doc = client.get_document(row['id'])
        text = doc.get('biblio', {}).get('ru', {}).get('title', '') + ' ' + doc.get('snippet', {}).get('description', '')
        examples.append(InputExample(texts=[row['query'], text]))
    model = SentenceTransformer(base_model, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    torch.mps.empty_cache()
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, output_path=output_path)
    return model


class HardTripletDataset(Dataset):
    def __init__(self, queries: List[str], positives: List[str], corpus: List[str], bm25_tokenized: List[List[str]], k_hard=5):
        self.queries = queries
        self.positives = positives
        self.corpus = corpus
        self.bm25 = BM25Okapi(bm25_tokenized)
        self.k_hard = k_hard

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        pos = self.positives[idx]
        scores = self.bm25.get_scores(q.split())
        topk = torch.topk(torch.tensor(scores), self.k_hard + 1).indices.tolist()
        hard_idxs = [i for i in topk if self.corpus[i] != pos][:self.k_hard]
        neg = self.corpus[hard_idxs[torch.randint(len(hard_idxs), (1,)).item()]]
        return q, pos, neg

def train_with_hard_negatives(
    model_name: str,
    queries: list[str],
    positives: list[str],
    corpus: list[str],
    tokenized_corpus: list[list[str]],
    batch_size: int = 1,
    accumulation_steps: int = 4,
    epochs: int = 5,
    lr: float = 2e-5,
    k_hard: int = 2
):
    model = SentenceTransformer(model_name, device=device)
    model._first_module().auto_model.gradient_checkpointing_enable()

    ds = HardTripletDataset(queries, positives, corpus, tokenized_corpus, k_hard)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=0.2)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (qs, ps, ns) in enumerate(loader, start=1):
            # tokenize
            bq = model.tokenizer(qs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            bp = model.tokenizer(ps, padding=True, truncation=True, return_tensors='pt', max_length=512)
            bn = model.tokenizer(ns, padding=True, truncation=True, return_tensors='pt', max_length=512)

            feats_q   = {k: v.to(device) for k, v in bq.items()}
            feats_pos = {k: v.to(device) for k, v in bp.items()}
            feats_neg = {k: v.to(device) for k, v in bn.items()}

            emb_q = model(feats_q)['sentence_embedding']
            emb_p = model(feats_pos)['sentence_embedding']
            emb_n = model(feats_neg)['sentence_embedding']

            loss = loss_fn(emb_q, emb_p, emb_n)
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps

            # only step & zero every N micro-batches
            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # catch any remaining gradients at epoch end
        if step % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")

    return model
