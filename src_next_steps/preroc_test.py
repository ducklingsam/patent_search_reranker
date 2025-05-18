import csv
import asyncio
import pandas as pd

from usage_example import preprocess_query

async def tpreprocessing(input_queries, out_path='preproc_results.csv'):
    results = []
    for q in input_queries:
        data = await preprocess_query(q)
        print(data)
        results.append({
            'original': q,
            'preprocessed': data['preprocessed_query'],
            'entities': ';'.join(data.get('entities', [])),
            'ipc_codes': ';'.join(data.get('ipc_codes', [])),
        })
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'Результаты сохранены в {out_path}')

if __name__ == '__main__':
    df = pd.read_csv('../data/manual_qrels.csv')
    queries = df['query'].unique()
    asyncio.run(tpreprocessing(queries))
