import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from db import get_query_fetcher
from eval import get_evaluator
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words


evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

train_data = pd.read_csv('train_data.csv')
train_data = train_data.fillna(0)

pivoted_data = train_data.pivot(index=['wms_sku_id'], columns=['picker_host'])
pivoted_data = pivoted_data.fillna(0)

re_idx = np.searchsorted(skus.wms_sku_id.values, pivoted_data.index.values)


feature_cols = []
for feature_name in ['task', 'pick', 'place']:
    for host in ['ropi01', 'ropi02', 'ropi03', 'ropi04']:
        n_map = np.zeros(skus.shape[0])
        s_map = np.zeros(skus.shape[0])

        s = pivoted_data[f'{feature_name}_success'][host].values
        n = pivoted_data[f'{feature_name}_total'][host].values

        s_map[re_idx] = s
        n_map[re_idx] = n

        feature_cols.extend([
            beta(a=s_map + 9, b=n_map - s_map + 1).mean(),
            beta(a=s_map + 9, b=n_map - s_map + 1).var(),
        ])

feature_cols = np.stack(feature_cols).T


sku_features = np.concatenate([
        np.stack(skus['dimensions_mm'].values),
        skus[['weight_g']].values,
        feature_cols], axis=1)


#scaler = StandardScaler()
#sku_features = scaler.fit_transform(sku_features)

dim_red = TSNE()
new_dim = dim_red.fit_transform(sku_features)

clustering = KMeans(n_clusters=16)
clustering.fit_transform(new_dim)

bic = evaluator.evaluate(
    skus=skus['wms_sku_id'].values,
    cluster_idx=clustering.labels_,
    metric='bic')

print(bic)
