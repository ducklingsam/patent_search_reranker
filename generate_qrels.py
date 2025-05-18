import os
import csv
import json
import uuid
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv
from src.rosclient import RosPatentClient

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ros = RosPatentClient()

TOP_K = 20  # сколько патентов показывать на один запрос

QUERY_SYSTEM_PROMPT = (
    "Ты — генератор реальных пользовательских патентных запросов. "
    "Сгенерируй краткий (3–7 слов) реальный осмысленный поисковый запрос на русском языке "
    "о технологиях, гаджетах, экологии, химии, энергетике, медицине или робототехнике. "
    "Не добавляй пояснений."
    "Под реальным имеется в виду - вероятно существующих патент"
)

def ask_for_query() -> str:
    """Получаем случайный запрос от GPT (можно расширить для подсказки тем)"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": "Сгенерируй запрос"},
        ],
        temperature=0.9,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip().strip("\n")

def search_patents(query: str):
    """Ищем патенты в RosPatent (top_k патентов)"""
    raw = ros.search_raw({"q": query, "limit": TOP_K, "offset": 0})
    return raw.get("hits", [])

def ensure_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query", "id", "label"])


def interactive_label(output_path: str):
    ensure_csv(output_path)
    session_id = uuid.uuid4()
    print("\n===== Интерактивная разметка qrels =====")
    print("Нажмите Enter для генерации нового запроса, 'q' — выйти.\n")

    while True:
        cmd = input("Enter/q > ").strip().lower()
        if cmd == "q":
            print("Выход.")
            break

        # 1. Генерируем запрос
        query = ask_for_query()
        print(f"\n\033[1mСгенерированный запрос:\033[0m {query}\n")

        # 2. Ищем патенты
        patents = search_patents(query)
        if not patents:
            print("\033[93mПатенты не найдены.\033[0m\n")
            continue

        rows = []
        for idx, p in enumerate(patents, 1):
            title = p.get("biblio", {}).get("ru", {}).get("title", "Нет названия")
            abstr = p.get("snippet", {}).get("description", "Нет описания")[:400]
            print(f"[{idx}] id={p['id']} | {title}\n    {abstr}\n")
            while True:
                lab = input("    Релевантен? 1/0/skip/stop > ").strip().lower()
                if lab in {"1", "0", "skip", "stop"}:
                    break
            if lab == "stop":
                print("\nСессия завершена.")
                return
            if lab == "skip":
                continue
            rows.append([query, p["id"], int(lab)])

        # 4. Записываем
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"\n✔ Сохранено {len(rows)} строк в {output_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Интерактивная генерация и разметка manual_qrels.csv"
    )
    parser.add_argument("--out", default="manual_qrels_extra.csv", help="Куда записывать разметку")
    args = parser.parse_args()

    interactive_label(args.out)
