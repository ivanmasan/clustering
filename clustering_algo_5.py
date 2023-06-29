import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from itertools import product
from evaluator.eval import Evaluator
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin


class PartialTransformer(TransformerMixin):
    def __init__(self, transformer, column_idx):
        self._columns = column_idx
        self._transformer = transformer

    def fit(self, X, y=None):
        filtered = X[:, self._columns]
        self._transformer.fit(filtered)
        return self

    def transform(self, X, y=None):
        ret = X.copy()
        filtered = ret[:, self._columns]
        ret[:, self._columns] = self._transformer.transform(filtered)
        return ret

    def inverse_transform(self, X, y=None):
        ret = X.copy()
        filtered = ret[:, self._columns]
        ret[:, self._columns] = self._transformer.inverse_transform(filtered)
        return ret

    def get_params(self, deep=False):
        return {
            'transformer': self._transformer,
            'column_idx': self._columns
        }


class LogarithmScaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X + 1)

    def inverse_transform(self, X, y=None):
        return np.exp(X) - 1

    def get_params(self, deep=False):
        return {}


class IQRClipper(TransformerMixin):
    def __init__(self, delta=3):
        self._delta = delta

    def fit(self, X, y=None):
        l, u = np.percentile(X, [25, 75], axis=0)
        self._lower = l - self._delta * (u - l)
        self._upper = u + self._delta * (u - l)
        return self

    def transform(self, X, y=None):
        return np.clip(X, a_min=self._lower, a_max=self._upper)

    def inverse_transform(self, X, y=None):
        return X

    def get_params(self, deep=False):
        return {'delta': self._delta}


softplus = torch.nn.Softplus()


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


def yield_repeat_batches(size: int, episodes: int = 1):
    idx = np.arange(size)
    for _ in tqdm(range(episodes)):
        yield np.random.choice(idx, size=size)


def entropy(x):
    x = x + 1e-20
    e = x * torch.log(x)
    return - e.sum()


class MetricTracker:
    def __init__(self):
        self._data = defaultdict(list)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self._data[key].append(value)

    def plot(self, key):
        plt.plot(self._data[key])


class AutogradClustering:
    def __init__(
            self,
            distance_decay,
            within_cluster_std,
            time_shift_std,
            entropy_reg_ratio,
            cluster_entropy_reg,
            central_cluster_reg,
            clusters,
            l2_reg,
            feature_names,
            transformer,
            X, y
    ) -> None:
        self._distance_decay = distance_decay
        self._within_cluster_std = within_cluster_std
        self._time_shift_std = time_shift_std
        self._entropy_reg_ratio = entropy_reg_ratio
        self._cluster_entropy_reg = cluster_entropy_reg
        self._central_cluster_reg = central_cluster_reg
        self._cluster_count = clusters
        self._l2_reg = l2_reg
        self._episodes = 0

        self._tracker = MetricTracker()

        self._samples = X.shape[0]
        self._features = X.shape[1]

        self._metrics = y.shape[3]
        self._metric_segments = y.shape[1]
        self._metric_time_segments = y.shape[2]

        self._feature_names = feature_names

        self._unscaled_X = X
        self._sc = transformer
        self._X = torch.tensor(transformer.fit_transform(X))
        self._y = torch.tensor(y)

        self._clusters = torch.Tensor(np.random.normal(0, 0.05, size=(self._cluster_count, self._features)))
        self._clusters.requires_grad = True

        self._p = torch.full(
            (self._cluster_count, self._metric_segments, self._metric_time_segments, self._metrics),
            -3.4, requires_grad=True
        )
        self._scale = torch.zeros((self._cluster_count, self._features), requires_grad=True)
        self._optimizer = torch.optim.SGD([self._clusters, self._p, self._scale], lr=0.01)

    def train(self, batch_size, episodes):
        for batch in yield_batches(self._samples, episodes=episodes, batch_size=batch_size):
            batch_y = self._y[batch]

            similarity = self._point_similarity(self._X[batch])
            similarity = similarity / similarity.sum(1, keepdim=True)

            avg_p = (similarity[:, :, None, None, None] * torch.sigmoid(self._p)[None]).sum(1)

            binom = torch.distributions.binomial.Binomial(
                total_count=batch_y[..., 1], probs=avg_p)
            data_log_lkhd = binom.log_prob(batch_y[..., 0]).sum()

            norm = torch.distributions.normal.Normal(
                loc=self._p.mean(1, keepdims=True), scale=self._within_cluster_std)
            model_log_lkhd = norm.log_prob(self._p).sum()

            time_diff = self._p[:, :, 1:] - self._p[:, :, :-1]
            norm = torch.distributions.normal.Normal(
                loc=0, scale=self._time_shift_std)
            time_shift_log_lkhd = norm.log_prob(time_diff).sum()

            self._p.std(1)

            dist_entropy = entropy(similarity) / similarity.shape[0]
            row_sim_sum = (similarity ** 2).sum(0)[:-1]
            row_sim_sum = row_sim_sum / row_sim_sum.sum()
            cluster_entropy = entropy(row_sim_sum)

            central_clusters_lkhd = torch.distributions.normal.Normal(
                loc=0, scale=self._central_cluster_reg).log_prob(self._clusters).sum()

            neg_log_lkhd = (- data_log_lkhd
                            - model_log_lkhd
                            - time_shift_log_lkhd
                            - central_clusters_lkhd
                            + self._entropy_reg_ratio * self._cluster_entropy_reg * dist_entropy
                            - (1 - self._entropy_reg_ratio) * self._cluster_entropy_reg * cluster_entropy
                            + self._l2_reg * (softplus(self._scale) ** 2).mean())

            if neg_log_lkhd.isnan():
                raise RuntimeError()

            self._optimizer.zero_grad()
            neg_log_lkhd.backward()
            torch.nn.utils.clip_grad_norm_([self._clusters, self._p, self._scale], 20)
            self._optimizer.step()

            self._tracker.log(
                data_log_lkhd=data_log_lkhd.item(),
                time_shift_log_lkhd=time_shift_log_lkhd.item(),
                model_log_lkhd=model_log_lkhd.item(),
                cluster_mean_dist=self.cluster_distance(),
                entropy=dist_entropy.item(),
                cluster_entropy=cluster_entropy.item(),
                log_lkhd=-neg_log_lkhd.item()
            )

            self._episodes += 1
            if self._episodes % 200 == 0:
                self._tracker.log(
                    cluster_sharpness=self.cluster_sharpness(),
                    cluster_count=self.cluster_count(),
                    edge_cluster_centres=self.edge_cluster_centres(),
                    within_cluster_dev=self.within_cluster_dev(),
                    sparsity=self.sparsity()
                )

    def plot(self, key):
        self._tracker.plot(key)

    def clusters(self, min_size=None):
        similarity = self._point_similarity(self._X)
        similarity = similarity.detach().numpy()
        inferred_clusters = np.argmax(similarity, axis=1)

        if min_size is None:
            return inferred_clusters

        item_count = Counter(inferred_clusters)
        mask = [item_count[i] <= min_size for i in range(self._cluster_count)]
        similarity[:, mask] = 0

        return np.argmax(similarity, axis=1)

    def relevant_clusters(self, min_size=None):
        clusters = self.clusters(min_size)
        return np.unique(clusters)

    def cluster_sharpness(self):
        similarity = self._point_similarity(self._X)
        similarity = similarity / similarity.sum(1, keepdim=True)
        return (torch.sort(similarity, dim=1)[0][:, -1] > 0.9).float().mean().item()

    def cluster_count(self):
        return len(Counter(self.clusters()))

    def cluster_distance(self):
        return torch.sqrt((self._clusters[self.relevant_clusters()] ** 2).sum(1)).mean().item()

    def within_cluster_dev(self):
        return self._p[self.relevant_clusters()].std(2).mean().item()

    def edge_cluster_centres(self):
        return (torch.abs(self._clusters[self.relevant_clusters()]) > 3).float().mean().item()

    def sparsity(self):
        return (softplus(self._scale[self.relevant_clusters()]) < 0.01).float().mean().item()

    def feature_importances(self, min_size=None):
        with torch.no_grad():
            clusters = self.clusters(min_size)
            relevant_clusters = np.unique(clusters)
            unscaled_clusters = self._sc.inverse_transform(self._clusters.detach().numpy())

            data = []

            for cluster in relevant_clusters:
                importances = self._scale[cluster]

                for idx in torch.flip(torch.argsort(importances), (0,)):
                    if importances[idx] < 4:
                        continue

                    data.append({
                        'cluster_id': cluster,
                        'importance': importances[idx].item(),
                        'feature': self._feature_names[idx],
                        'centre_value': unscaled_clusters[cluster, idx],
                        'z_value': self._clusters[cluster, idx].item(),
                        'percentile': (self._X[:, idx] < self._clusters[cluster, idx]).float().mean().item()
                    })
        return pd.DataFrame(data)

    def _point_similarity(self, X):
        difference = X[:, None, :] - self._clusters[None, :, :]

        difference = difference * softplus(self._scale)[None, :, :]
        distance = (difference ** 2).sum(2)
        distance = torch.exp(-distance / self._distance_decay)

        return distance

    def similarities(self):
        return self._point_similarity(self._X)


data = np.load('query_data/data.npz', allow_pickle=True)
feats = np.load('query_data/X.npz', allow_pickle=True)
text_emb = np.load(f'text_clusters/cluster_tsne_2.npz')

y = data['data']
train_y = y[:, :, :-2]
valid_y = y[:, :, -2:]

valid_y = valid_y.sum(2)
X = np.hstack([feats['data'][:, 3:], text_emb['emb']])

feature_names = list(feats['feature_names'][3:]) + [f'text_cluster_{k}' for k in range(text_emb['emb'].shape[1])]

#X = np.hstack([X, X[:, 5:6] / X[:, 3:4]])
#feature_names.append('density')

X = np.hstack([X, np.product(X[:, 1:3], axis=1, keepdims=True)])
feature_names.append('exposed_surface')


pipeline = Pipeline([
    ('log_scaler', PartialTransformer(LogarithmScaler(), [3, 4, 5])),
#    ('clipper', IQRClipper(3)),
    ('scaler', StandardScaler())
])

evaluator = Evaluator(
    targets=valid_y,
    skus=data['skus'].astype(int),
    sku_meta_data=pd.read_csv('query_data/skus.csv')
)

clustering = AutogradClustering(
    distance_decay=40,
    within_cluster_std=0.5,
    time_shift_std=0.3,
    entropy_reg_ratio=0.3,
    cluster_entropy_reg=5,
    central_cluster_reg=0.8,
    clusters=48,
    l2_reg=1,
    feature_names=feature_names,
    transformer=pipeline,
    X=X,
    y=train_y
)



clustering.train(128, 5000)


clustering.plot('cluster_count')
clustering.plot('cluster_sharpness')
clustering.plot('edge_cluster_centres')
clustering.plot('within_cluster_dev')
clustering.plot('sparsity')

clustering.plot('log_lkhd')
clustering.plot('entropy')
clustering.plot('cluster_entropy')
clustering.plot('cluster_mean_dist')
clustering.plot('data_log_lkhd')
clustering.plot('model_log_lkhd')
clustering.plot('time_shift_log_lkhd')

imp = clustering.feature_importances()

output_path = Path('results/ver20')
lkhd = evaluator.overview(
    output_path=output_path,
    image_path=Path('images'),
    skus=data['skus'].astype(int),
    cluster_idx=clustering.clusters(16),
    features=X,
    feature_names=feature_names
)


imp.to_excel(output_path / 'importances.ods')
similarity = clustering.similarities()
np.savez(output_path / 'similarities.npz', similarity)

most_similiar_clusters = pd.DataFrame({'sku': data['skus'].astype(int)})
most_similiar_clusters[['b5', 'b4', 'b3', 'b2', 'b1']] = np.argsort(similarity, axis=1)[:, -5::]
most_similiar_clusters[['s5', 's4', 's3', 's2', 's1']] = np.sort(similarity, axis=1)[:, -5::]
most_similiar_clusters.to_excel(output_path / 'cluster_similarities.ods', index=False)


raise
self = clustering
similarity = self._point_similarity(self._X).detach()
similarity = similarity / similarity.sum(1, keepdim=True)
similarity = similarity.numpy()

min_size = 16
item_count = Counter(np.argmax(similarity, axis=1))
mask = [item_count[i] <= min_size for i in range(self._cluster_count)]
similarity[:, mask] = 0
similarity = similarity / similarity.sum(1, keepdims=True)

plt.hist(np.sort(similarity, axis=1)[:, -1])



self._p
self._y.shape

binom = torch.distributions.binomial.Binomial(
    total_count=self._y[:, None, ..., 1].detach(),
    probs=torch.sigmoid(self._p)[None].detach())
data_log_lkhd = binom.log_prob(self._y[:, None, ..., 0])
data_log_lkhd = data_log_lkhd.sum((2, 3, 4))
data_log_lkhd = data_log_lkhd.numpy()

torch.argmax(data_log_lkhd, axis=1)

1761633
1386926
939798
939804
939810
1468377
1468397
1468402
idx = np.where(np.isin(skus, [1468377, 1468397, 1468402]))[0]
similarity[idx][:, [21, 29, 46]]


idx = np.where(skus == 1468402)[0][0]

print(self._y[idx][:, :, -1, 1].sum())

re_idx = np.argsort(data_log_lkhd[idx])[-5:]
print(re_idx)
print(data_log_lkhd[idx, re_idx])


re_idx = np.argsort(similarity[idx])[-10:]
print(re_idx)
print(similarity[idx, re_idx])


clusters = self.clusters()
sorted_sim = np.sort(similarity, axis=1)

mean_belonging = np.zeros(self._cluster_count)
for i in np.unique(clusters):
    mask = clusters == i
    belong = sorted_sim[mask][:, -2]
    mean_belonging[i] = belong.mean()



distance = np.sqrt(((self._X[:, None, :] - self._X[None, :, :]) ** 2).sum(2))
same_cluster = clusters[:, None] == clusters[None, :]
idx_x, idx_y = np.where((distance < 0.1) & (~same_cluster))
skus[idx_x]
skus[idx_y]



def entropy(x):
    x = x + 1e-20
    e = x * np.log(x)
    return - e.sum()


def reg(similarity, ratio):
    e = entropy(similarity) / similarity.shape[0]  # low

    row_sim_sum = (similarity ** 2).sum(0)[:-1]
    row_sim_sum = row_sim_sum / row_sim_sum.sum()
    ce = entropy(row_sim_sum)  # high

    return ratio * e - (1 - ratio) * ce  # low


3000
20 * 128

SAMPLES = 128
FEATURE_COUNT = 10
RATIO = 1

similarity = np.zeros((SAMPLES, FEATURE_COUNT))
similarity[:, 0] = 1
print("One Cluster", reg(similarity, RATIO))

similarity = np.zeros((SAMPLES, FEATURE_COUNT))
similarity[:, :2] = 0.5
print("Two Cluster", reg(similarity, RATIO))

similarity = np.zeros((SAMPLES, FEATURE_COUNT))
similarity[:] = 1 / FEATURE_COUNT
print("Random", reg(similarity, RATIO))


for b in [0.3, 0.5, 0.7, 0.9, 1]:
    similarity = np.zeros((SAMPLES, FEATURE_COUNT))
    similarity[:] = (1 - b) / (FEATURE_COUNT - 1)
    similarity[np.arange(SAMPLES), np.arange(SAMPLES) % FEATURE_COUNT] = b
    print(f"Even {b} split", reg(similarity, RATIO))


for b in [0.3, 0.5, 0.7, 0.9, 1]:
    similarity = np.zeros((SAMPLES, FEATURE_COUNT))
    similarity[:] = (1 - b) / (FEATURE_COUNT - 2)
    similarity[np.arange(SAMPLES), np.arange(SAMPLES) % FEATURE_COUNT] = b / 2
    similarity[np.arange(SAMPLES), (1 + np.arange(SAMPLES)) % FEATURE_COUNT] = b / 2
    print(f"Two {b} split", reg(similarity, RATIO))


for b in [0.3, 0.5, 0.7, 0.9, 1]:
    similarity = np.zeros((SAMPLES, FEATURE_COUNT))
    similarity[:] = (1 - b) / (FEATURE_COUNT - 1)
    split_val = int(SAMPLES / 2)
    similarity[np.arange(split_val), 0] = b
    similarity[np.arange(SAMPLES)[split_val:], np.arange(SAMPLES)[split_val:] % FEATURE_COUNT] = b
    print(f"Half Majority {b} split", reg(similarity, RATIO))
