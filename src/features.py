from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

from .rosclient import RosPatentClient
from .utils import device, download_model

client = RosPatentClient()

_model_path = os.getenv('EMBED_MODEL_PATH', 'models/contrastive-rubert')
if os.path.isdir(_model_path):
    embed_model = SentenceTransformer(_model_path, device=device)
else:
    download_model('https://limewire.com/d/gbCI2#mQvObSb6Mz', 'models/contrastive-rubert')
_query_ipc_cache = {}


def fetch_bm25_score(query: str, patent_id: str) -> float:
    resp = client.search(query, limit=1, filter={"id": patent_id})
    if resp:
        return float(resp[0].get('similarity_norm', 0.0))
    return 0.0


def infer_query_ipc(query: str, top_k: int = 10, top_n: int = 3) -> list:
    if query in _query_ipc_cache:
        return _query_ipc_cache[query]
    hits = client.search(query, limit=top_k)
    all_ipc = []
    for p in hits:
        ipc = client.get_document(p['id']).get('biblio', {}).get('ru', {}).get('ipc', [])
        all_ipc.extend(ipc)
    top_codes = [code for code, _ in Counter(all_ipc).most_common(top_n)]
    _query_ipc_cache[query] = top_codes
    return top_codes


def compute_dense_similarity(query: str, text: str) -> float:
    q_emb = embed_model.encode(query, convert_to_tensor=False)
    d_emb = embed_model.encode(text, convert_to_tensor=False)
    return float(cosine_similarity([q_emb], [d_emb])[0][0])


def compute_ipc_similarity(query: str, patent_id: str) -> float:
    q_codes = set(infer_query_ipc(query))
    p_codes = set(client.get_document(patent_id).get('biblio', {}).get('ru', {}).get('ipc', []))
    if not q_codes or not p_codes:
        return 0.0
    return len(q_codes & p_codes) / len(q_codes | p_codes)


def extract_features(query: str, patents: list) -> list:
    feats = []
    for p in patents:
        pid = p.get('id')
        snippet = p.get('snippet', {})
        text = snippet.get('description', '') if isinstance(snippet, dict) else snippet
        feats.append({
            'id': pid,
            'bm25': fetch_bm25_score(query, pid),
            'dense_sim': compute_dense_similarity(query, text),
            'ipc_sim': compute_ipc_similarity(query, pid),
            'year': p.get('year')
        })
    return feats
