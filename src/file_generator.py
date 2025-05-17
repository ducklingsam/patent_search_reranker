from rosclient import RosPatentClient
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util



def extract_xml_text(xml_string: str) -> str:
    try:
        soup = BeautifulSoup(xml_string, 'lxml-xml')
        return ' '.join(el.get_text(strip=True) for el in soup.find_all(['p', 'span']))
    except Exception as e:
        print(f"⚠️ Ошибка при парсинге XML: {e}")
        return ''




def all_patents_generator():
    client = RosPatentClient()
    labels = pd.read_csv('../data/labeled_patents.csv')
    unique_ids = labels['id'].unique()
    records = []
    for pid in unique_ids:
        doc = client.get_document(patent_id=pid)

        title = str(doc.get('title', '') or '')

        abstract = doc.get('abstract', '')
        if isinstance(abstract, dict):
            abstract = abstract.get('ru', '')
        abstract_text = extract_xml_text(abstract)

        claims = doc.get('claims', {})
        if isinstance(claims, dict):
            claims_text = extract_xml_text(claims.get('ru', ''))
        else:
            claims_text = extract_xml_text(str(claims))

        full_text = ' '.join([title, abstract_text, claims_text])
        records.append({'id': pid, 'text': full_text})

    df_corpus = pd.DataFrame(records)
    df_corpus.to_csv('../data/all_patents.csv', index=False)


def top_candidates():
    labels = pd.read_csv('../data/labeled_patents.csv')
    print("Столбцы в labeled_patents.csv:", labels.columns.tolist())

    queries_df = labels[['topic', 'query']].drop_duplicates().reset_index(drop=True)
    print("Найдено запросов:", len(queries_df))

    corpus = pd.read_csv('../data/all_patents.csv')
    print("Столбцы в all_patents.csv:", corpus.columns.tolist(), "Всего документов:", len(corpus))

    model = SentenceTransformer('../models/rubert_patent_embedder_hardneg')

    corpus_texts = corpus['text'].tolist()
    corpus_ids = corpus['id'].tolist()
    corpus_emb = model.encode(corpus_texts,
                              batch_size=32,
                              convert_to_tensor=True,
                              show_progress_bar=True)

    top_k = 50
    rows = []
    for _, row in queries_df.iterrows():
        topic = row['topic']
        qtext = row['query']
        q_emb = model.encode(qtext, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, corpus_emb)[0]
        top = scores.topk(top_k)
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            rows.append({
                'topic': topic,
                'query': qtext,
                'id': corpus_ids[idx],
                'bi_score': score
            })

    candidates = pd.DataFrame(rows)
    candidates = candidates.merge(
        labels[['topic', 'id', 'label']],
        on=['topic', 'id'],
        how='left'
    )
    candidates['label'] = candidates['label'].fillna(0).astype(int)

    candidates.to_csv('../data/query_candidates_top50.csv', index=False)
    print("Готово! Сгенерирован файл query_candidates_top50.csv")


if __name__ == "__main__":
    top_candidates()
