import pandas as pd, random
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 1. Данные
cands  = pd.read_csv('../data/query_candidates_top50.csv')
corpus = pd.read_csv('../data/all_patents.csv')[['id','text']]
text_map = dict(zip(corpus['id'], corpus['text']))

# 2. Примеры (sampling 1:10)
examples = []
for topic, grp in cands.groupby('topic'):
    pos = grp[grp['label']==1]
    neg = grp[grp['label']==0]
    for _, p in pos.iterrows():
        q, tp = p['query'], text_map[p['id']]
        examples.append(InputExample(texts=[q, tp], label=1.0))
        for _, n in neg.sample(min(len(neg), 10), random_state=42).iterrows():
            tn = text_map[n['id']]
            examples.append(InputExample(texts=[q, tn], label=0.0))
print("Примеры:", len(examples))

# 3. Split
train_s, dev_s = train_test_split(examples, test_size=0.2, random_state=42)
train_loader = DataLoader(train_s, shuffle=True, batch_size=16)

# 4. Модель-регрессор
ce = CrossEncoder(
    'DeepPavlov/rubert-base-cased',
    num_labels=1,
    max_length=256,
    device='mps'
)

from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

# 1. Build the evaluator from your dev set
dev_evaluator = CEBinaryClassificationEvaluator.from_input_examples(
    dev_s,
    name='patent-dev-reg'    # just a label for saving results
)

# 2. Fine-tune, passing the evaluator instead of dev_samples
ce.fit(
    train_dataloader=train_loader,
    evaluator=dev_evaluator,
    epochs=2,
    evaluation_steps=100,
    output_path='../models/crossencoder_patent_reg',
    save_best_model=True,
)
