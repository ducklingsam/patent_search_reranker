from src.rosclient import RosPatentClient

client = RosPatentClient()
print(len(client.search("тестовый запрос", limit=5)))