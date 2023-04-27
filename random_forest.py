from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import beta
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from unidecode import unidecode

from db import get_query_fetcher
from eval import get_evaluator, create_cluster_images
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words


evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])


train_data = pd.read_csv('train_data.csv')
train_data = train_data.fillna(0)
mask = train_data['pick_total'] > 0
train_data = train_data[mask]

dimensions = np.stack(skus['dimensions_mm'].values)


sku_features = np.concatenate([
    dimensions,
    dimensions.min(1, keepdims=True),
    dimensions.max(1, keepdims=True),
    dimensions.max(1, keepdims=True) / dimensions.min(1, keepdims=True),
    skus[['weight_g']].values], axis=1)

ohe = OneHotEncoder(sparse=False)
host = ohe.fit_transform(train_data[['picker_host']])

sku_to_train_idx = np.searchsorted(skus['wms_sku_id'], train_data['wms_sku_id'])
X = sku_features[sku_to_train_idx]
X = np.hstack([X, host])

success_y = train_data['pick_success'].values
total_y = train_data['pick_total'].values
y = success_y / total_y
weights = 1 / beta(a=success_y + 1, b=total_y - success_y + 1).var()

model = RandomForestRegressor(
    max_leaf_nodes=2,
    n_estimators=10,
)
model.fit(X, y, sample_weight=weights)

cluster_by_host = []
for i in range(4):
    zeros = np.zeros((sku_features.shape[0], 4))
    zeros[:, i] = 1
    cluster_by_host.append(
        model.apply(
            np.concatenate([sku_features, zeros], axis=1)
        )
    )

leafs = np.concatenate(cluster_by_host, axis=1)

cluster_ohe = OneHotEncoder(sparse=False)
features = cluster_ohe.fit_transform(leafs)

clustering = KMeans(n_clusters=16)
clustering.fit_transform(features)


bic = evaluator.evaluate(
    skus=skus.wms_sku_id.values,
    cluster_idx=clustering.labels_,
    metric='bic'
)

print(bic)

create_cluster_images(
    skus=skus.wms_sku_id.values,
    cluster_idx=clustering.labels_,
    output_path=Path('clusters/random_forest')
)
