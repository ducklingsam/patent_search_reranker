import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from lime.lime_tabular import LimeTabularExplainer
from concurrent.futures import ThreadPoolExecutor
from .reranker import PatentReranker
from .features import extract_features
from .rosclient import RosPatentClient
from .utils import device

print(f"Using device: {device}")

CACHE_FILE = "patent_cache.json"

def load_cache():
    """Загрузка кэша результатов поиска."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Сохранение кэша."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def explain_shap(model, X_sample, output_path='shap_summary.png'):
    """SHAP: визуализация важности признаков с использованием MPS."""
    model = model.to(device)
    explainer = shap.TreeExplainer(model)

    X_subset = X_sample.sample(n=min(5, len(X_sample)), random_state=42)
    X_subset_tensor = torch.tensor(X_subset.values, dtype=torch.float32).to(device)

    shap_values = explainer.shap_values(X_subset_tensor.cpu().numpy())

    shap.summary_plot(shap_values, X_subset, feature_names=X_subset.columns, show=False)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def explain_lime(model, X_train, feature_names, instance, output_path='lime_explanation.html'):
    """LIME: объяснение для конкретного примера."""
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='regression'
    )
    exp = explainer.explain_instance(instance.values, model.predict, num_features=5)
    exp.save_to_file(output_path)

def process_query(q, client, reranker, output_dir, cache):
    """Обработка одного запроса: извлечение признаков, объяснение SHAP и LIME."""
    try:
        if q in cache:
            patents = cache[q]
        else:
            patents = client.search(q, limit=20)
            cache[q] = patents
            save_cache(cache)

        feats = extract_features(q, patents)
        X = pd.DataFrame(feats)[['bm25', 'dense_sim', 'ipc_sim']]

        shap_path = os.path.join(output_dir, f'shap_{q.replace(" ", "_")}.png')
        explain_shap(reranker.model, X, output_path=shap_path)

        instance = X.iloc[0]
        lime_path = os.path.join(output_dir, f'lime_{q.replace(" ", "_")}.html')
        explain_lime(reranker.model, X, X.columns, instance, output_path=lime_path)

    except Exception as e:
        print(f"Error processing query '{q}': {e}")

def analyze_explainability(manual_csv, reranker_model_path, output_dir='explainability', max_workers=8):
    """Анализ интерпретируемости с распараллеливанием."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(manual_csv)
    queries = df['query'].unique()

    client = RosPatentClient()
    reranker = PatentReranker(reranker_model_path)

    cache = load_cache()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for q in queries:
            executor.submit(process_query, q, client, reranker, output_dir, cache)

    print(f"Explainability results saved in {output_dir}")
