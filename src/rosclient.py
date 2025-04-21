import os
import requests
from dotenv import load_dotenv

load_dotenv()

class RosPatentClient:
    """
    Клиент для работы с RosPatent API v0.2
    Пример использования:
    ```python
        client = RosPatentClient()
        hits = client.search("искусственный интеллект в медицине", limit=10)
    ```
    """

    BASE_URL = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ROS_PATENT_API_KEY')
        if not self.api_key:
            raise RuntimeError("ROS_PATENT_API_KEY не задан в окружении")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def search(self, query: str, limit: int = 10, offset: int = 0, datasets: list = None) -> list:
        """Выполняет поиск по запросу.
        :param query: Строка запроса.
        :param limit: Максимальное количество результатов (по умолчанию 10).
        :param offset: Смещение для пагинации (по умолчанию 0).
        :param datasets: Список наборов данных для поиска (по умолчанию используются все доступные):
            ["ru_till_1994", "ru_since_1994", "cis", "dsgn_ru", "ap", "cn", "ch", "au", "gb",
             "kr", "ca", "at", "jp", "ep", "de", "fr", "pct", "us", "dsgn_kr", "dsgn_cn",
             "dsgn_jp", "others"].
        :return: Список патентов (словарей).
        :rtype: list[dict]
        """
        payload = {
            "q": query,
            "limit": limit,
            "offset": offset,
            "datasets": datasets or ["ru_till_1994","ru_since_1994","cis","dsgn_ru","ap","cn","ch","au","gb","kr","ca","at","jp","ep","de","fr","pct","us","dsgn_kr","dsgn_cn","dsgn_jp","others"]
        }
        resp = self.session.post(f"{self.BASE_URL}/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", [])

    def get_document(self, patent_id: str) -> dict:
        resp = self.session.get(f"{self.BASE_URL}/docs/{patent_id}")
        resp.raise_for_status()
        return resp.json()
