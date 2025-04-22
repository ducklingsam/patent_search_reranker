import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from .reranker import PatentReranker
from .features import extract_features
from .rosclient import RosPatentClient


def explain_shap(model, X_sample, output_path='shap_summary.png'):
    """SHAP: визуализация важности признаков."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Визуализация
    shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return shap_values


def explain_lime(model, X_train, feature_names, instance, output_path='lime_explanation.html'):
    """LIME: объяснение для конкретного примера."""
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode='regression'
    )
    exp = explainer.explain_instance(instance.values, model.predict[0], num_features=len(feature_names))
    exp.save_to_file(output_path)
    return exp


def analyze_explainability(manual_csv, reranker_model_path, output_dir='explainability'):
    """Анализ интерпретируемости для всех запросов."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(manual_csv)
    client = RosPatentClient()
    reranker = PatentReranker(reranker_model_path)

    for q in df['query'].unique():
        patents = client.search(q, limit=20)
        feats = extract_features(q, patents)
        X = pd.DataFrame(feats)[['bm25', 'dense_sim', 'ipc_sim']]

        # SHAP
        shap_values = explain_shap(reranker.model, X, output_path=f'{output_dir}/shap_{q.replace(" ", "_")}.png')

        # LIME для первого патента
        instance = X.iloc[0]
        explain_lime(
            reranker.model,
            X,
            X.columns,
            instance,
            output_path=f'{output_dir}/lime_{q.replace(" ", "_")}.html'
        )

    print(f"Explainability results saved in {output_dir}")