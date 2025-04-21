import os
import pandas as pd
from src.rosclient import RosPatentClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = RosPatentClient()
oclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = oclient.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """Сгенерируй 10 конкретных тем для патентов. Ориентируй запросы под рынок РФ и стран СНГ. Запрос должен быть очелевочен. Верни результат в формате JSON, где каждая тема — это объект с полями "topic" (название темы) и "query" (поисковый запрос для патентного поиска). Пример:
            [
                {"topic": "Искусственный интеллект в медицине", "query": "ИИ для диагностики заболеваний"},
                ...
            ]
            Не добавляй ничего лишнего, только объект JSON!. Убедись, что темы разнообразные и актуальные."""
        }
    ],
    temperature=0.7,
    max_tokens=500
)

import json
topics = json.loads(response.choices[0].message.content.replace('```json', '').replace('```', ''))

labeled_data = []

for item in topics:
    topic = item['topic']
    query = item['query']
    print(f"\nТема: {topic}")
    print(f"Запрос: {query}")

    try:
        patents = client.search(query, limit=10)
    except Exception as e:
        print(f"Ошибка при поиске для запроса '{query}': {e}")
        continue

    for patent in patents:
        pid = patent.get('id', 'unknown')
        title = patent.get('biblio', {}).get('ru', {}).get('title', 'Нет названия')
        snippet = patent.get('snippet', {}).get('description', 'Нет описания')

        print("\nПатент:")
        print(f"ID: {pid}")
        print(f"Название: {title}")
        print(f"Описание: {snippet}")

        while True:
            label = input("Релевантен? (1/0): ")
            if label in ['1', '0']:
                label = int(label)
                break
            print("Пожалуйста, введите 1 или 0")

        labeled_data.append({
            'topic': topic,
            'query': query,
            'id': pid,
            'label': label
        })

df = pd.DataFrame(labeled_data)
df.to_csv('labeled_patents.csv', index=False)
print("\nРазмеченные данные сохранены в 'labeled_patents.csv'")