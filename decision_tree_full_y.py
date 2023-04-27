from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import beta
from sklearn.tree import DecisionTreeRegressor

from db import get_query_fetcher
from eval import get_evaluator, create_cluster_image_folders, create_cluster_summary_images, prepare_folder
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

train_data = pd.read_csv('train_data.csv')
train_data = train_data.fillna(0)

dimensions = np.stack(skus['dimensions_mm'].values)

sku_features = np.concatenate([
    dimensions,
    dimensions.prod(1, keepdims=True),
    dimensions.min(1, keepdims=True),
    dimensions.max(1, keepdims=True),
    dimensions.max(1, keepdims=True) / dimensions.min(1, keepdims=True),
    skus[['weight_g']].values], axis=1)

ohe = OneHotEncoder(sparse=False)
host = ohe.fit_transform(train_data[['picker_host']])

sku_to_train_idx = np.searchsorted(skus['wms_sku_id'], train_data['wms_sku_id'])
X = sku_features[sku_to_train_idx]
# X = np.hstack([X, host])

success_y = train_data[['pick_success', 'place_success', 'task_success']].values
total_y = train_data[['pick_total', 'place_total', 'task_total']].values
y = success_y / total_y
weights = 1 / beta(a=success_y + 1, b=total_y - success_y + 1).std()

nan_mask = np.isnan(y)
weights[nan_mask] = 0.9
y[nan_mask] = 0

model = DecisionTreeRegressor(max_leaf_nodes=16, min_samples_leaf=30)
model.fit(X, y, sample_weight=weights.mean(1))

clusters = model.apply(sku_features)

lkhd = evaluator.evaluate(
    skus=skus.wms_sku_id.values,
    cluster_idx=clusters,
    metric='log_lkhd'
)
print(lkhd)  # 22335.54


evaluator.overview(
    output_path=Path('results/scikit_tree'),
    image_path=Path('small_images'),
    skus=skus.wms_sku_id.values,
    cluster_idx=clusters,
    features=sku_features,
    feature_names=['X_dim', 'Y_dim', 'Z_dim', 'area',
                   'min_dim', 'max_dim', 'dim_ratio', 'weight']
)


tree.plot_tree(
    model, fontsize=12,
    node_ids=True,
    feature_names=["X_dim", 'Y_dim', 'Z_dim', 'area',
                   'min_dim', 'max_dim', 'dim_ratio', 'weight']
)
