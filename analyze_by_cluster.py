from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt


class DailyVisualizer:
    def __init__(self, daily_failures: np.ndarray, dates: np.ndarray,
                 clusters: np.ndarray, metric_names: np.ndarray, picker_names: np.ndarray,
                 cluster_metadata: pd.DataFrame):
        # Shape: Skus, Pickers, Days, Metrics, Failure/Success
        assert len(clusters) == daily_failures.shape[0]
        assert len(picker_names) == daily_failures.shape[1]
        assert len(dates) == daily_failures.shape[2]
        assert len(metric_names) == daily_failures.shape[3]
        assert daily_failures.shape[4] == 2

        self._daily_failures = daily_failures
        self._dates = dates
        self._metric_names = metric_names
        self._picker_names = picker_names

        self._clustered_daily_data = self._create_clustered_failure_data(daily_failures, clusters)
        assert self._clustered_daily_data.shape[0] == len(cluster_metadata)
        assert np.all(cluster_metadata.cluster.values == self._clusters)

        self._superclusters = cluster_metadata.supercluster.values
        self._supercluster_names = cluster_metadata.superclustername.values
        self._cluster_names = (cluster_metadata.superclustername + ' - ' + cluster_metadata.name).values

    def _create_clustered_failure_data(self, daily_failures, clusters):
        clustered_failures = np.zeros((len(np.unique(clusters)), *daily_failures.shape[1:]))
        self._clusters = np.unique(labels)

        for i, cluster_id in enumerate(self._clusters):
            mask = labels == cluster_id
            clustered_failures[i] = daily_failure_data[mask].sum(0, keepdims=True)

        return clustered_failures

    def _filter(self, item_names, items):
        if items is None:
            return np.arange(len(item_names))

        if isinstance(items, list):
            items = np.array(items)

        if items.dtype.kind == 'i':
            return items
        elif items.dtype.kind == 'U':
            ret = []
            for item in items:
                idx = np.where(item_names == item)[0]
                if len(idx) == 0:
                    raise ValueError(f"Value {item} not found in {item_names}")
                ret.append(idx[0])
            return np.array(ret)
        else:
            raise RuntimeError("Wrong index type")

    def _filter_hosts(self, data, pickers):
        idx = self._filter(self._picker_names, pickers)
        return data[:, idx].sum(1, keepdims=True)

    def _filter_days(self, data, dates, start_date, end_date):
        mask = np.full(shape=len(dates), fill_value=True)

        if start_date is not None:
            mask = (dates >= start_date) & mask

        if end_date is not None:
            mask = (dates <= end_date) & mask

        return data[:, :, mask], dates[mask]

    def _filter_metrics(self, data, metrics):
        idx = self._filter(self._metric_names, metrics)
        return data[:, :, :, idx], self._metric_names[idx]

    def _squash_clusters(self, data, cluster_ids):
        if cluster_ids is None:
            return data, self._cluster_names

        if isinstance(cluster_ids, str) and cluster_ids == 'superclusters':
            cluster_ids = np.unique(self._superclusters)

        if isinstance(cluster_ids, list):
            cluster_ids = np.array(cluster_ids)

        assert isinstance(cluster_ids, np.ndarray)

        grouped_clusters = np.empty(len(self._clusters), dtype=int)
        assigned = np.full(shape=len(self._clusters), fill_value=False)
        names = []

        latest_idx = 0

        for cluster_id in cluster_ids:
            if cluster_id in self._superclusters:
                mask = cluster_id == self._superclusters
                name = self._supercluster_names[mask][0]
            else:
                mask = cluster_id == self._clusters
                name = self._cluster_names[mask][0]

            assigned[mask] = True
            grouped_clusters[mask] = latest_idx
            latest_idx += 1
            names.append(name)

        grouped_clusters[~assigned] = latest_idx + np.arange((~assigned).sum())
        names.extend(self._cluster_names[~assigned])
        latest_idx += np.sum(~assigned)

        new_data = np.zeros((latest_idx, *data.shape[1:]))

        assert np.max(grouped_clusters) + 1 == len(np.unique(grouped_clusters))

        for old_index, new_index in enumerate(grouped_clusters):
            new_data[new_index] += data[old_index]

        return new_data, np.array(names)

    def plot_failure_rates(self, hosts=None, metrics=None,
                           start_date=None, end_date=None):
        data = self._filter_hosts(self._clustered_daily_data, hosts)
        data, dates = self._filter_days(data, self._dates, start_date, end_date)
        data, metrics = self._filter_metrics(data, metrics)
        data = data.sum((0, 1))
        s = data[:, :, 0]
        n = data[:, :, 1]

        p = s / n
        p[np.isnan(p)] = 0

        plt.plot(dates, p, linewidth=4)
        plt.xticks(dates, rotation=90, size=15)
        plt.yticks(size=15)
        plt.grid()
        plt.legend(metrics)
        plt.show()

    def plot_cluster_normalized_failure_rates(self, hosts=None, metrics=None,
                                              start_date=None, end_date=None):
        data = self._clustered_daily_data
        data = self._filter_hosts(data, hosts)
        data, dates = self._filter_days(data, self._dates, start_date, end_date)
        data, metrics = self._filter_metrics(data, metrics)
        data = data.sum(1)
        s = data[:, :, :, 0]
        n = data[:, :, :, 1]

        global_ratios = n.sum(1, keepdims=True)
        global_ratios = global_ratios / global_ratios.sum(0)

        p = s / n
        nan_mean = np.nanmean(p, axis=1)
        for i in range(p.shape[2]):
            for j in range(p.shape[0]):
                p[j, :, i][np.isnan(p[j, :, i])] = nan_mean[j, i]

        adjusted_p = (p * global_ratios).sum(0)
        original_p = s.sum(0) / n.sum(0)

        plt.plot(dates, original_p, '--', linewidth=5)
        plt.plot(dates, adjusted_p, linewidth=5)
        plt.xticks(dates, rotation=90, size=15)
        plt.yticks(size=15)
        plt.grid()
        plt.legend(list(metrics) + [f'Normalized {x}' for x in metrics])
        plt.show()

    def plot_cluster_contribution(self, metric, hosts=None,
                                  start_date=None, end_date=None,
                                  cluster_ids=None):
        data = self._clustered_daily_data
        data, cluster_names = self._squash_clusters(data, cluster_ids)
        data = self._filter_hosts(data, hosts)
        data, dates = self._filter_days(data, self._dates, start_date, end_date)
        data, metric = self._filter_metrics(data, [metric])
        data = data.sum((1, 3))
        s = data[:, :, 0]
        n = data[:, :, 1]

        global_ratios = n.sum(1, keepdims=True)
        global_ratios = global_ratios / global_ratios.sum(0)

        contribs = (s / n) * global_ratios
        contribs[np.isnan(contribs)] = 0
        idx = np.argsort(contribs.std(1))[::-1]

        bottom_line = np.zeros(s.shape[1])
        for i in idx:
            plt.fill_between(dates, bottom_line, bottom_line + contribs[i])
            bottom_line += contribs[i]

        plt.xticks(dates, rotation=90, size=15)
        plt.yticks(size=15)
        plt.grid()
        plt.legend(cluster_names[idx])

    def plot_picker_normalized_failure_rates(self, metrics=None, start_date=None, end_date=None):
        data = self._clustered_daily_data
        data, dates = self._filter_days(data, self._dates, start_date, end_date)
        data, metrics = self._filter_metrics(data, metrics)
        data = data.sum(0)
        s = data[:, :, :, 0]
        n = data[:, :, :, 1]

        global_ratios = n.sum(1, keepdims=True)
        global_ratios = global_ratios / global_ratios.sum(0)

        p = s / n
        nan_mean = np.nanmean(p, axis=1)
        for i in range(p.shape[2]):
            for j in range(p.shape[0]):
                p[j, :, i][np.isnan(p[j, :, i])] = nan_mean[j, i]

        adjusted_p = (p * global_ratios).sum(0)
        original_p = s.sum(0) / n.sum(0)

        plt.plot(dates, original_p, '--', linewidth=5)
        plt.plot(dates, adjusted_p, linewidth=5)
        plt.xticks(dates, rotation=90, size=15)
        plt.yticks(size=15)
        plt.grid()
        plt.legend(list(metrics) + [f'Normalized {x}' for x in metrics])
        plt.show()

    def plot_picker_contribution(self, metric, start_date=None, end_date=None):
        data = self._clustered_daily_data
        data, dates = self._filter_days(data, self._dates, start_date, end_date)
        data, metric = self._filter_metrics(data, [metric])
        data = data.sum((0, 3))
        s = data[:, :, 0]
        n = data[:, :, 1]

        global_ratios = n.sum(1, keepdims=True)
        global_ratios = global_ratios / global_ratios.sum(0)

        contribs = (s / n) * global_ratios
        contribs[np.isnan(contribs)] = 0
        idx = np.argsort(contribs.std(1))[::-1]

        bottom_line = np.zeros(s.shape[1])
        for i in idx:
            plt.fill_between(dates, bottom_line, bottom_line + contribs[i])
            bottom_line += contribs[i]

        plt.xticks(dates, rotation=90, size=15)
        plt.yticks(size=15)
        plt.grid()
        plt.legend(self._picker_names[idx])


# From database/daily_data_pull.py
daily_failures = np.load('query_data/daily_failure_data.npz', allow_pickle=True)
daily_failure_data = daily_failures['data']

clusters = np.load('results/lkhd_tree_new/clusters.npz')
labels = clusters['labels']
label_skus = clusters['skus']

daily_skus = daily_failures['skus']
m = np.isin(daily_skus, label_skus.astype(str))
daily_failure_data = daily_failure_data[m]

cluster_meta_data = pd.read_csv('results/lkhd_tree_new/cluster_summary.csv')

viz = DailyVisualizer(
    daily_failures=daily_failure_data,
    picker_names=daily_failures['hosts'],
    metric_names=np.array(['Place Fail', 'No Pick Attempts', 'Detect', '3C', 'Total']),
    dates=daily_failures['days'],
    clusters=labels,
    cluster_metadata=cluster_meta_data
)


viz.plot_failure_rates(start_date=dt.date(2023, 4, 9))
viz.plot_cluster_normalized_failure_rates(start_date=dt.date(2023, 4, 9))
viz.plot_cluster_contribution(start_date=dt.date(2023, 4, 9), metric='Total')
viz.plot_picker_normalized_failure_rates(start_date=dt.date(2023, 4, 9))
viz.plot_picker_contribution(start_date=dt.date(2023, 4, 9), metric='Total')

viz.plot_cluster_contribution(start_date=dt.date(2023, 4, 9), metric='Place Fail',
                              cluster_ids=[101, 39, 45, 46])

viz.plot_cluster_contribution(start_date=dt.date(2023, 4, 9), metric='Total', cluster_ids=[101, 39, 45, 46])
viz.plot_picker_contribution(start_date=dt.date(2023, 4, 9), metric='Total')

viz.plot_cluster_contribution(start_date=dt.date(2023, 4, 9), metric='Total', cluster_ids=[101, 39, 45, 46], hosts=['ropi03'])
