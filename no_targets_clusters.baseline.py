from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from db import get_query_fetcher
from eval import get_evaluator, create_cluster_images
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words


evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

sku_features = np.concatenate([
        np.stack(skus['dimensions_mm'].values),
        skus[['weight_g']].values], axis=1)


scaler = StandardScaler()
sku_features = scaler.fit_transform(sku_features)

dim_red = TSNE()
new_dim = dim_red.fit_transform(sku_features)

clustering = KMeans(n_clusters=16)
clustering.fit_transform(new_dim)

bic = evaluator.evaluate(
    skus=skus['wms_sku_id'].values,
    cluster_idx=clustering.labels_,
    metric='bic')

print(bic)

names = evaluator.names(
    skus=skus['wms_sku_id'].values,
    cluster_idx=clustering.labels_,
    verbose=True)

plt.scatter(*new_dim.T, c=clustering.labels_)

create_cluster_images(
    skus=skus.wms_sku_id.values,
    cluster_idx=clustering.labels_,
    output_path=Path('clusters/no_target_clusters'),
    summary_image=True,
)
