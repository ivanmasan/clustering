from copy import deepcopy
from pathlib import Path
from scipy.stats import binom
from collections import Counter
import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import beta, binom
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from scipy.stats import boxcox
from db import get_query_fetcher
from eval import get_evaluator
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words

import torch


softplus = torch.nn.Softplus()


def point_similarity(clusters, scale, X):
    difference = X[:, None, :] - clusters[None, :, :]
    difference = difference * softplus(scale)[None, :, :]
    distance = (difference ** 2).sum(2)
    distance = torch.exp(-distance / 10)

    return distance


def p_vector(similarity, success_y, total_y):
    weights = similarity / similarity.sum(1, keepdim=True)

    p = (((weights * success_y).sum(0) / weights.sum(0))
         / ((weights * total_y).sum(0) / weights.sum(0)))

    return p


def yield_batches(size: int, batch_size: int, episodes: int = 1):
    for _ in tqdm(range(episodes)):
        idx = np.arange(size)
        np.random.shuffle(idx)

        while len(idx):
            yield idx[:batch_size]
            idx = idx[batch_size:]


evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

sku_text_embedding = np.load('text_clusters.npz')['arr_0']


train_data = pd.read_csv('train_data.csv')
train_data = train_data.fillna(0)
mask = train_data['pick_total'] > 0
train_data = train_data[mask]

dimensions = np.stack(skus['dimensions_mm'].values)

sku_features = np.concatenate([
    dimensions,
    dimensions.prod(1, keepdims=True),
    dimensions.min(1, keepdims=True),
    dimensions.max(1, keepdims=True),
    dimensions.max(1, keepdims=True) / dimensions.min(1, keepdims=True),
    skus[['weight_g']].values,
    sku_text_embedding], axis=1)


#for i in range(sku_features.shape[1]):
#    sku_features[:, i] = boxcox(sku_features[:, i])[0]


sku_to_train_idx = np.searchsorted(skus['wms_sku_id'], train_data['wms_sku_id'])
unscaled_X = sku_features[sku_to_train_idx]

sc = StandardScaler()
X = sc.fit_transform(unscaled_X)
scaled_sku_features = sc.transform(sku_features)

success_y = train_data['pick_success'].values.reshape(-1, 1)
total_y = train_data['pick_total'].values.reshape(-1, 1)

X = torch.tensor(X)

success_y = torch.tensor(success_y)
total_y = torch.tensor(total_y)

SAMPLES = X.shape[0]
FEATURES = X.shape[1]
CLUSTERS = 16

clusters = torch.Tensor(np.random.normal(0, 0.05, size=(CLUSTERS, FEATURES)))
clusters.requires_grad = True

p = torch.zeros(CLUSTERS, requires_grad=True)
scale = torch.zeros((CLUSTERS, FEATURES), requires_grad=True)
optimizer = torch.optim.SGD([clusters, p, scale], lr=0.01)

cluster_history = []

i = 0
for batch in yield_batches(SAMPLES, batch_size=256, episodes=5000):
    similarity = point_similarity(clusters, scale, X[batch])
    similarity = similarity / similarity.sum(1, keepdim=True)

    avg_p = (similarity * torch.sigmoid(p)).sum(1)

    binom = torch.distributions.binomial.Binomial(total_count=total_y[batch].flatten(), probs=avg_p)
    neg_log_lkhd = - binom.log_prob(success_y[batch].flatten()).sum()

    if neg_log_lkhd.isnan():
        raise RuntimeError()

    optimizer.zero_grad()
    neg_log_lkhd.backward()
    torch.nn.utils.clip_grad_norm_([clusters, p, scale], 20)
    optimizer.step()

    i += 1
    if i % 1000 == 0:
        cluster_history.append(deepcopy(clusters.detach()))

        print(torch.sigmoid(p))

        similarity = point_similarity(clusters, scale, torch.tensor(scaled_sku_features))
        inference_clusters = torch.argmax(similarity, dim=1).numpy()
        print(Counter(inference_clusters))

        log_lkhd = evaluator.evaluate(
            skus=skus.wms_sku_id.values,
            cluster_idx=inference_clusters,
            metric='log_lkhd'
        )

        print(log_lkhd)

        similarity = point_similarity(clusters, scale, X)
        similarity = similarity / similarity.sum(1, keepdim=True)

        avg_p = (similarity * torch.sigmoid(p)).sum(1)

        binom = torch.distributions.binomial.Binomial(total_count=total_y.flatten(), probs=avg_p)
        log_lkhd = - binom.log_prob(success_y.flatten()).sum()
        print(log_lkhd)


evaluator.overview(
    output_path=Path('results/custom_clustering_16_2'),
    image_path=Path('small_images'),
    skus=skus.wms_sku_id.values,
    cluster_idx=inference_clusters,
    features=sku_features,
    feature_names=(['X_dim', 'Y_dim', 'Z_dim', 'area',
                    'min_dim', 'max_dim', 'dim_ratio', 'weight']
                   + [f'text_cluster_{i}' for i in range(8)])
)
