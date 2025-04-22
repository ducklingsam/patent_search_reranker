import pandas as pd
import lightgbm as lgb
from .features import extract_features
from .rosclient import RosPatentClient


class PatentReranker:
    """
    Обучение и применение LightGBM LambdaRank ранкера для патентного поиска
    """
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path:
            self.model = lgb.Booster(model_file=model_path)

    def train(self, gold_csv: str, output_path: str):
        # Загрузка разметки
        df = pd.read_csv(gold_csv)
        client = RosPatentClient()
        rows = []
        groups = []
        # Формируем обучающий набор
        for q in df['query'].unique():
            sub = df[df['query'] == q]
            patents = client.search(q, limit=len(sub))
            feats = extract_features(q, patents)
            for f in feats:
                pid = f['id']
                # метка релевантности для данного патента
                label_series = sub[sub['id'] == pid]['label']
                label = int(label_series.iloc[0]) if not label_series.empty else 0
                rows.append({**f, 'label': label, 'query': q})
            groups.append(len(patents))
        data = pd.DataFrame(rows)
        # Признаки и целевая
        X = data[['bm25', 'dense_sim', 'ipc_sim']]
        y = data['label']
        # Создаём Dataset для LambdaRank
        train_data = lgb.Dataset(X, y, group=groups)
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 1   # снизили минимальное число данных в листе для небольших групп
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        model.save_model(output_path)
        self.model = model
        return model

    def predict(self, query: str, top_k: int = 10, features: list = None):
        if self.model is None:
            raise ValueError('Модель не загружена.')
        client = RosPatentClient()
        patents = client.search(query, limit=top_k)
        feats = extract_features(query, patents)
        df_feats = pd.DataFrame(feats)

        if features is None:
            features = ['bm25', 'dense_sim', 'ipc_sim']

        df_feats['score'] = self.model.predict(df_feats[features])
        return df_feats.sort_values('score', ascending=False), patents