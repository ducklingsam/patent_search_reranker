import os
import requests
import time
from dotenv import load_dotenv
from loguru import logger

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

    def search(self, query: str, limit: int = 10, offset: int = 0, datasets: list = None, tries: int = 3,
               filter: dict = None) -> list:
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
        for try_ in range(tries):
            try:
                payload = {
                    "q": query,
                    "limit": limit,
                    "offset": offset,
                    "datasets": datasets or ["ru_till_1994","ru_since_1994","cis","dsgn_ru","ap","cn","ch","au","gb","kr",
                                             "ca","at","jp","ep","de","fr","pct","us","dsgn_kr","dsgn_cn","dsgn_jp","others"],
                }
                if filter is not None:
                    payload["filter"] = filter
                resp = self.session.post(f"{self.BASE_URL}/search", json=payload)
                resp.raise_for_status()
                logger.info(f"Successfully searched for {query}")
                data = resp.json()
                return data.get("hits", [])
            except Exception as e:
                logger.warning(f'Error while searching: {e}. Retrying {try_ + 1}/{tries} in 1 sec')
                time.sleep(1)
        resp.raise_for_status()

    def get_document(self, patent_id: str, tries: int = 3) -> dict:
        """
        Fetches the document details of a given patent from a remote API.

        This method sends a GET request to the API to retrieve the document information
        for the given patent ID. It retries the request up to the specified number of
        attempts in case of request errors. If all attempts fail, an exception is
        raised. The result is returned as a dictionary containing the document details.

        Parameters:
            patent_id (str): The unique identifier for the patent whose document needs
                to be fetched.
            tries (int): The number of retries for the request in case of failure.
                Defaults to 3.

        Returns:
            dict: A dictionary containing the details of the requested patent document.

        Raises:
            Exception: If all retry attempts for the GET request fail.
        """
        for try_ in range(tries):
            resp = self.session.get(f"{self.BASE_URL}/docs/{patent_id}")
            try:
                resp.raise_for_status()
                logger.info(f"Fetched document {patent_id}")
                return resp.json()
            except Exception as e:
                logger.warning(f'Error while getting document {patent_id}: {e}. Retrying {try_ + 1}/{tries} in 1 sec')
                time.sleep(1)
        resp.raise_for_status()


    def search_raw(self, payload: dict, tries: int = 3):
        for try_ in range(tries):
            try:
                resp = self.session.post(f"{self.BASE_URL}/search", json=payload)
                resp.raise_for_status()
                logger.info(f"Successfully searched for {payload}")
                data = resp.json()
                return data
            except Exception as e:
                logger.warning(f'Error while searching: {e}. Retrying {try_ + 1}/{tries} in 1 sec')
                time.sleep(1)
