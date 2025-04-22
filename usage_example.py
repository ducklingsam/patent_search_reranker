from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
import uuid
from src.reranker import PatentReranker
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

reranker = PatentReranker(model_path="models/patent_reranker.txt")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class ChatRequest(BaseModel):
    chat_id: str | None = None
    text: str | None = None
    chat_history: list[dict] | None = None

class ChatResponse(BaseModel):
    chat_id: str
    text: str


async def generate_answer(text: str, patents: list, chat_history: list[dict] = None):
    system_prompt = "Ты патентный помощник. Используй информацию по найденным патентам, чтобы ответить на запрос."
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.extend({"role": "user", "content": text + f"Найденные патенты: {patents}"})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if request.chat_id is None:
            request.chat_id = str(uuid.UUID(version=4))
        df_feats, patents = reranker.predict(request.text)
        response = []
        for _, row in df_feats.iterrows():
            patent = next(p for p in patents if p['id'] == row['id'])
            response.append({
                "id": row['id'],
                "title": patent.get('biblio', {}).get('ru', {}).get('title', 'Нет названия'),
                "description": patent.get('snippet', {}).get('description', 'Нет описания')[:200],
                "score": float(row['score'])
            })
        answer = await generate_answer(request.text, response, request.chat_history)

        return ChatResponse(text=answer, chat_id=request.chat_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
