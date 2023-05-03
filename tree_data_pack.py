import numpy as np
import pandas as pd
import torch

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

sku_text_embedding = np.load('text_clusters.npz')['arr_0']
categorical = np.argmin(sku_text_embedding, axis=1, keepdims=True)

train_data = pd.read_csv('train_data.csv')
train_data = train_data.fillna(0)

s_cols = ['task_success', 'pick_success', 'place_success']
n_cols = ['task_total', 'pick_total', 'place_total']

train_data = train_data.pivot(
    index=['wms_sku_id'], columns=['picker_host'],
    values=s_cols + n_cols).fillna(0)


sku_to_train_idx = np.searchsorted(skus['wms_sku_id'], train_data.index)

s_targets = np.zeros((skus.shape[0], len(n_cols), train_data[n_cols[0]].shape[1]))
n_targets = np.zeros((skus.shape[0], len(n_cols), train_data[n_cols[0]].shape[1]))

for i, (s_col, n_col) in enumerate(zip(s_cols, n_cols)):
    s_targets[sku_to_train_idx, i] = train_data[s_col].values
    n_targets[sku_to_train_idx, i] = train_data[n_col].values

s_targets = torch.tensor(s_targets)
n_targets = torch.tensor(n_targets)

dimensions = np.stack(skus['dimensions_mm'].values)

X = np.concatenate([
    dimensions,
    dimensions.prod(1, keepdims=True),
    dimensions.min(1, keepdims=True),
    dimensions.max(1, keepdims=True),
    dimensions.max(1, keepdims=True) / dimensions.min(1, keepdims=True),
    skus[['weight_g']].values], axis=1)

payload = {
    'numerical': X,
    'categorical': categorical,
    'n': n_targets,
    's': s_targets
}

np.savez("tree_data.npz", **payload)








