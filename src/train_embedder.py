import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from .rosclient import RosPatentClient
from .embedder import train_with_hard_negatives

labeled_path = 'data/labeled_patents.csv'
queries, positive_ids = [], []
with open(labeled_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            queries.append(row['query'])
            positive_ids.append(row['id'])

all_ids = set()
all_ids.update(positive_ids)

all_ids = list(all_ids)

client = RosPatentClient()


def fetch_text(patent_id: str) -> tuple[str, str]:
    doc = client.get_document(patent_id)

    def extract_text(field, lang='ru'):
        """Безопасно извлекает текст из словаря или строки."""
        if isinstance(field, dict):
            return field.get(lang, '') or ''
        return field or ''

    biblio = doc.get("biblio", {})
    biblio_ru = biblio.get("ru", {})

    title = extract_text(biblio_ru.get("title", ""))
    abstract = extract_text(biblio_ru.get("abstract", ""))

    claims_raw = doc.get("claims", [])
    if isinstance(claims_raw, dict):
        claims_text = extract_text(claims_raw.get("text", ""))
    elif isinstance(claims_raw, list):
        claims_text = " ".join(claims_raw)
    elif isinstance(claims_raw, str):
        claims_text = claims_raw
    else:
        claims_text = ""

    full_text = " ".join([title, abstract, claims_text])
    return patent_id, full_text

corpus_texts = {}
with ThreadPoolExecutor(max_workers=8) as exe:
    futures = [exe.submit(fetch_text, pid) for pid in all_ids]
    for future in as_completed(futures):
        pid, text = future.result()
        corpus_texts[pid] = text

corpus = [corpus_texts[pid] for pid in all_ids]
tokenized_corpus = [text.split() for text in corpus]

positives = [corpus_texts[pid] for pid in positive_ids]

model = train_with_hard_negatives(
    model_name='ai-forever/sbert_large_nlu_ru',
    queries=queries,
    positives=positives,
    corpus=corpus,
    tokenized_corpus=tokenized_corpus,
    batch_size=4,
    epochs=5,
    lr=2e-5,
    k_hard=5,
    accumulation_steps=8
)

model.save('models/rubert_patent_embedder_hardneg')
