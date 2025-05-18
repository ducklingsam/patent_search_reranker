from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
import uuid
import json
import pandas as pd
from src.reranker import PatentReranker
from src.rosclient import RosPatentClient
from src.features import extract_features
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Порог минимального скорора релевантности
THRESHOLD = 0.2
# Максимальное число кандидатов для поиска
TOP_K = 100

# Инициализация клиентов
reranker = PatentReranker(model_path="models/patent_reranker.txt")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ros_client = RosPatentClient()

class ChatRequest(BaseModel):
    chat_id: str | None = None
    text: str
    chat_history: list[dict] | None = []

class ChatResponse(BaseModel):
    chat_id: str
    chat_history: list[dict] | None = None
    text: str

async def preprocess_query(query: str) -> str:
    system_prompt = (
        "Ты — помощник по патентному поиску."
        " Преобразуй пользовательский запрос в формальный патентный стиль, в именительном падеже (для ElasticSearch)."
        " Верни только преобразованную строку без дополнительных данных."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=100
        )
        preprocessed = resp.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print("Preprocess error:", e)
        preprocessed = query
    return preprocessed

async def expand_query(preprocessed_query: str) -> list:
    system_prompt = (
        "Ты — патентный помощник."
        " Для данного формализованного запроса предложи 2–3 синонимичные формулировки."
        " Верни только список расширений в формате JSON-массива строк."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": preprocessed_query}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=100
        )
        raw = resp.choices[0].message.content
        content = raw.replace('```json', '').replace('```', '')
        expansions = json.loads(content)
    except Exception as e:
        print("Expand error:", e)
        expansions = []
    return expansions

async def generate_answer(
    original_query: str,
    preprocessed_query: str,
    expansions: list,
    patents: list,
    chat_history: list[dict] | None = None
) -> str:
    system_prompt = "Ты — патентный помощник. Используй найденные патенты для ответа."
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Оригинал запроса: {original_query}"})
    messages.append({"role": "user", "content": f"Обработанный запрос: {preprocessed_query}"})
    if expansions:
        messages.append({"role": "user", "content": f"Расширения: {', '.join(expansions)}"})
    messages.append({"role": "user", "content": f"Найденные патенты: {patents}"})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return resp.choices[0].message.content

@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if request.chat_id is None:
            request.chat_id = str(uuid.uuid4())

        # 1. Препроцессинг запроса
        preprocessed = await preprocess_query(request.text)

        # 2. Расширение запроса
        expansions = await expand_query(preprocessed)

        # 3. Построение комбинированного поиска
        phrases = [preprocessed] + expansions
        combined_q = " OR ".join([f"({p})" for p in phrases])

        # 4. Поиск через RosPatent Raw API без фильтров
        raw = ros_client.search_raw({
            "q": combined_q,
            "limit": TOP_K,
            "offset": 0
        })
        patents = raw.get("hits", [])

        # 5. Ранжирование списка патентов
        feats = extract_features(preprocessed, patents)
        df_feats = pd.DataFrame(feats)
        if not df_feats.empty:
            features = [c for c in df_feats.columns if c not in ('id', 'year')]
            df_feats['score'] = reranker.model.predict(df_feats[features])
            df_feats = df_feats.sort_values('score', ascending=False)

        # 6. Фолбэк если нет релевантных
        if df_feats.empty or df_feats['score'].iloc[0] < THRESHOLD:
            fallback = (
                "К сожалению, по вашему запросу патенты не найдены. "
                "Можете уточнить область или технологию?"
            )
            history = request.chat_history.copy() if request.chat_history else []
            history.append({"role": "user", "content": request.text})
            history.append({"role": "assistant", "content": fallback})
            return ChatResponse(text=fallback, chat_id=request.chat_id, chat_history=history)

        # 7. Подготовка списка для генерации ответа
        response_list = []
        for _, row in df_feats.iterrows():
            patent = next((p for p in patents if p['id'] == row['id']), {})
            response_list.append({
                "id": row['id'],
                "title": patent.get('biblio', {}).get('ru', {}).get('title', 'Нет названия'),
                "description": patent.get('snippet', {}).get('description', 'Нет описания')[:200],
                "score": float(row['score'])
            })

        # 8. Генерация ответа GPT
        answer = await generate_answer(
            request.text,
            preprocessed,
            expansions,
            response_list,
            request.chat_history
        )

        # Обновляем историю чата
        history = request.chat_history.copy() if request.chat_history else []
        history.append({"role": "user", "content": request.text})
        history.append({"role": "assistant", "content": answer})

        return ChatResponse(text=answer, chat_id=request.chat_id, chat_history=history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
