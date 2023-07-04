from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from clearml import Logger


softplus = torch.nn.Softplus()


def p_vector(similarity, success_y, total_y):
    weights = similarity / similarity.sum(1, keepdim=True)

    p = (((weights * success_y).sum(0) / weights.sum(0))
         / ((weights * total_y).sum(0) / weights.sum(0)))

    return p


def yield_batches(size: int, batch_size: int):
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
        self._iteration = 0
        self._batch_data = defaultdict(float)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self._data[key].append(value)
            logger = Logger.current_logger()
            if logger is not None:
                logger.report_scalar(
                    key, "", iteration=self._iteration, value=value
                )

    def batch_log(self, **kwargs):
        for key, value in kwargs.items():
            self._batch_data[key] += value

    def plot(self, key):
        plt.plot(self._data[key])
        plt.title(key)
        plt.show()

    def __getitem__(self, item):
        return self._data[item]

    def new_iteration(self):
        self.log(**self._batch_data)
        self._batch_data = defaultdict(float)
        self._iteration += 1


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
            X, y,
            evaluator=None,
            sku_list=None
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

        self._evaluator = evaluator
        self._sku_list = sku_list

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
        for _ in tqdm(range(episodes)):
            for batch in yield_batches(self._samples, batch_size=batch_size):
                self._train_batch(batch)

            if self._episodes % 10 == 0:
                self._log_state_info()

            self._episodes += 1
            self._tracker.new_iteration()
        self._log_state_info()

    def _log_state_info(self):
        self._tracker.log(
            cluster_sharpness=self.cluster_sharpness(),
            cluster_count=self.cluster_count(),
            edge_cluster_centres=self.edge_cluster_centres(),
            within_cluster_dev=self.within_cluster_dev(),
            sparsity=self.sparsity(),
            cluster_mean_dist=self.cluster_distance(),
        )
        if self._evaluator is not None:
            self.evaluate()

    def _train_batch(self, batch):
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

        self._tracker.batch_log(
            data_log_lkhd=data_log_lkhd.item(),
            time_shift_log_lkhd=time_shift_log_lkhd.item(),
            model_log_lkhd=model_log_lkhd.item(),
            entropy=dist_entropy.item(),
            cluster_entropy=cluster_entropy.item(),
            log_lkhd=-neg_log_lkhd.item()
        )

    def plot(self, key):
        self._tracker.plot(key)

    @property
    def tracker(self):
        return self._tracker

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

    def evaluate(self, min_size=None):
        lkhd = self._evaluator.evaluate(
            skus=self._sku_list,
            cluster_idx=self.clusters(min_size),
        )
        self._tracker.log(evaluation=lkhd)
        return lkhd
