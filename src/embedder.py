import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

from .rosclient import RosPatentClient
from .utils import device


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