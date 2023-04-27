from __future__ import annotations

import graphviz
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Literal, List

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from itertools import combinations
from scipy.stats import betabinom

from eval import get_evaluator


data = np.load("tree_data.npz", allow_pickle=True)
s = data.get('s')
n = data.get('n')
numerical = data.get('numerical')
categorical = data.get('categorical')


@dataclass
class Split:
    variable_type: Literal['numerical', 'categorical']
    variable_idx: int
    split_value: Union[float, set]
    mask: np.ndarray

    def __repr__(self):
        return f'({self.variable_type} idx={self.variable_idx}, value={self.split_value})'


@dataclass
class Node:
    id: float
    lkhd: float
    sample_size: np.ndarray = None
    p: np.ndarray = None
    item_size: int = None
    split: Optional[Split] = None
    split_improvement: Optional[float] = None
    left_child: Optional[Node] = None
    right_child: Optional[Node] = None
    parent: Optional[Node] = None
    parent_way: Optional[Literal['right', 'left']] = None

    def __repr__(self):
        return f'Node(id={self.id}, split_improvement={self.split_improvement}, split=({self.split}))'

    def is_leaf(self) -> bool:
        return self.split is None and self.parent is not None

    def is_path_node(self) -> bool:
        return self.split is not None

    def is_orphan(self) -> bool:
        return self.split is None and self.parent is None

    def is_removable(self) -> bool:
        return (
            not self.is_leaf()
            and self.left_child.is_leaf()
            and self.right_child.is_leaf()
        )

    def turn_to_leaf(self) -> None:
        if self.is_leaf():
            return

        if not self.left_child.is_leaf():
            self.left_child.turn_to_leaf()

        if not self.right_child.is_leaf():
            self.right_child.turn_to_leaf()

        self.left_child.parent = None
        self.right_child.parent = None

        self.left_child = None
        self.right_child = None
        self.split_improvement = None
        self.split = None


def _batch_log_lkhd(n, s):
    p = s.sum(0, keepdims=True) / n.sum(0, keepdims=True)
    p[np.isnan(p)] = 0
    return stats.binom(n=n, p=p).logpmf(s).sum()


class Tree:
    def __init__(
        self,
        minimal_improvement: float,
        max_leaves: int,
        min_cluster_size: int,
        numerical_proposition_count: int,
        max_items_per_node: int,
        refine_candidates: int,
        numerical_feature_names: List[str] = None,
        categorical_feature_names: List[str] = None,
    ):
        self._nodes = []
        self._nodes_dict = {}
        self._node_count = 0
        self._max_leaves = max_leaves
        self._max_items_per_node = max_items_per_node
        self._min_split_improvement = minimal_improvement
        self._min_cluster_size = min_cluster_size
        self._refine_candidates = refine_candidates
        self._numerical_proposition_count = numerical_proposition_count
        self._numerical_feature_names = numerical_feature_names
        self._categorical_feature_names = categorical_feature_names

    def _new_node(self, lkhd):
        node = Node(self._node_count, lkhd)
        self._nodes.append(node)
        self._nodes_dict[self._node_count] = node
        self._node_count += 1
        return node

    def fit(self, numerical, categorical, n, s):
        self._assign_feature_names(numerical, categorical)
        self._data_points = n.shape[0]
        self._root = self._new_node(_batch_log_lkhd(n, s))
        self._assign_children(self._root, numerical, categorical, n, s)
        self._prune()
        return self._labels()

    def _assign_feature_names(self, numerical, categorical):
        if self._numerical_feature_names is None:
            self._numerical_feature_names = list(np.arange(numerical.shape[1]))

        if self._categorical_feature_names is None:
            self._categorical_feature_names = list(np.arange(categorical.shape[1]))

        assert len(self._numerical_feature_names) == numerical.shape[1]
        assert len(self._categorical_feature_names) == categorical.shape[1]

    def _assign_children(self, node, numerical, categorical, n, s):
        node.sample_size = n.sum((0, 2))
        node.p = s.sum((0, 2)) / node.sample_size
        node.item_size = len(n)

        if numerical.shape[0] <= self._min_cluster_size * 2:
            return

        split, (left_lkhd, right_lkhd) = self._find_best_split(numerical, categorical, n, s)
        if split is None:
            return

        split_improvement = left_lkhd + right_lkhd - node.lkhd
        if split_improvement < self._min_split_improvement:
            return

        node.split = split
        node.left_child = self._new_node(left_lkhd)
        node.left_child.parent = node
        node.left_child.parent_way = 'left'

        node.right_child = self._new_node(right_lkhd)
        node.right_child.parent = node
        node.right_child.parent_way = 'right'

        node.split_improvement = split_improvement

        self._assign_children(node.left_child, numerical[split.mask],
                              categorical[split.mask], n[split.mask], s[split.mask])
        self._assign_children(node.right_child, numerical[~split.mask],
                              categorical[~split.mask], n[~split.mask], s[~split.mask])

    def _yield_numerical_split_propositions(self, x):
        sort_idx = np.argsort(x)

        if len(x) < self._min_cluster_size * 2:
            return np.array([])

        split_idx_values = np.linspace(
            self._min_cluster_size,
            len(x) - self._min_cluster_size,
            self._numerical_proposition_count
        ).astype(int)

        for split_idx in split_idx_values:
            mask = np.full(shape=x.shape, fill_value=False)
            mask[sort_idx[:split_idx]] = True
            yield x[sort_idx[split_idx]], mask

    def _yield_categorical_split_propositions(self, x):
        group_counter = Counter(x)
        masks = np.zeros((x.size, x.max() + 1))
        masks[np.arange(x.size), x] = 1

        categories = set(group_counter.keys())
        total_items = sum(group_counter.values())

        for i in range(np.floor(len(categories) / 2).astype(int)):
            for subset in combinations(categories, i + 1):
                subset = set(subset)

                subset_items = sum([group_counter[s] for s in subset])
                if self._min_cluster_size <= subset_items <= total_items - self._min_cluster_size:
                    yield (
                        subset,
                        sum([masks[:, s] for s in subset]).astype(bool)
                    )

    def _yield_all_splits(self, numerical, categorical):
        for i in range(numerical.shape[1]):
            generator = self._yield_numerical_split_propositions(numerical[:, i])

            for split_value, mask in generator:
                yield Split(
                    variable_type='numerical',
                    variable_idx=i,
                    split_value=split_value,
                    mask=mask
                )

        for i in range(categorical.shape[1]):
            generator = self._yield_categorical_split_propositions(categorical[:, i])

            for set_value, mask in generator:
                yield Split(
                    variable_type='categorical',
                    variable_idx=i,
                    split_value=set_value,
                    mask=mask
                )

    def _refine_split(self, split, numerical):
        x = numerical[:, split.variable_idx]
        sort_idx = np.argsort(x)

        idx = np.searchsorted(x[sort_idx], split.split_value)

        step_size = (len(x) - 2 * self._min_cluster_size) / self._numerical_proposition_count
        step_size = np.floor(step_size).astype(int)

        if step_size < 2:
            return

        split_idx_values = np.arange(
            np.clip(idx - step_size, self._min_cluster_size, len(x) - self._min_cluster_size),
            np.clip(idx + step_size, self._min_cluster_size, len(x) - self._min_cluster_size)
        )

        if len(split_idx_values) == 0:
            return

        values = x[sort_idx[split_idx_values]]

        duplicates_mask = values[1:] != values[:-1]
        duplicates_mask = np.insert(duplicates_mask, 0, True)

        split_idx_values = split_idx_values[duplicates_mask]

        for split_idx in split_idx_values:
            mask = np.full(shape=x.shape, fill_value=False)
            mask[sort_idx[:split_idx]] = True
            yield Split(
                variable_type='numerical',
                variable_idx=split.variable_idx,
                split_value=x[sort_idx[split_idx]],
                mask=mask
            )

    def _find_best_split(self, numerical, categorical, n, s):
        splits = []
        lkhd = []
        lkhd_tuple = []

        for split in self._yield_all_splits(numerical, categorical):
            log_lkhd = (
                _batch_log_lkhd(n[split.mask], s[split.mask]),
                _batch_log_lkhd(n[~split.mask], s[~split.mask])
            )
            sum_log_lkhd = sum(log_lkhd)

            splits.append(split)
            lkhd.append(sum_log_lkhd)
            lkhd_tuple.append(log_lkhd)

        if self._refine_candidates > 0:
            for idx in np.argsort(lkhd)[-self._refine_candidates:]:
                if splits[idx].variable_type != 'numerical':
                    continue

                for split in self._refine_split(splits[idx], numerical):
                    log_lkhd = (
                        _batch_log_lkhd(n[split.mask], s[split.mask]),
                        _batch_log_lkhd(n[~split.mask], s[~split.mask])
                    )
                    sum_log_lkhd = sum(log_lkhd)

                    splits.append(split)
                    lkhd.append(sum_log_lkhd)
                    lkhd_tuple.append(log_lkhd)

        idx = np.argmax(lkhd)
        return splits[idx], lkhd_tuple[idx]

    def _leaf_nodes(self):
        for node in self._nodes:
            if node.is_leaf():
                yield node

    def _prune(self):
        is_removable_node = np.full(shape=self._node_count, fill_value=False)
        improvement = np.empty(shape=self._node_count)

        for node in self._nodes:
            improvement[node.id] = node.split_improvement
            is_removable_node[node.id] = node.is_removable() and node.item_size < self._max_items_per_node

        leaf_count = len(list(self._leaf_nodes()))

        while leaf_count > self._max_leaves:
            idx = np.argmin(improvement[is_removable_node])
            node_id = np.where(is_removable_node)[0][idx]

            is_removable_node[node_id] = False
            node = self._nodes_dict[node_id]

            node.turn_to_leaf()

            if node.parent.item_size < self._max_items_per_node:
                if node.parent_way == 'left':
                    if node.parent.right_child.split is None:
                        is_removable_node[node.parent.id] = True
                elif node.parent_way == 'right':
                    if node.parent.left_child.split is None:
                        is_removable_node[node.parent.id] = True

            leaf_count -= 1

    def _labels(self):
        labels = np.empty(self._data_points, dtype=int)
        for node in self._leaf_nodes():
            leaf_id = node.id

            decision_path = []
            while node.parent is not None:
                decision_path.append(node)
                node = node.parent
            decision_path.append(node)
            decision_path = decision_path[::-1]

            idx = np.arange(self._data_points)
            for parent, child in zip(decision_path[:-1], decision_path[1:]):
                mask = parent.split.mask
                if child.parent_way == 'right':
                    mask = ~mask
                idx = idx[mask]

            labels[idx] = leaf_id

        return labels

    def visualize_tree(self, output_folder: Optional[Path] = None, view: bool = False):
        if output_folder is None:
            output_folder = Path('temp')
            output_folder.mkdir(exist_ok=True)

        tree_viz = graphviz.Digraph('tree', comment='TREE')

        for node in self._nodes:
            if node.is_orphan():
                continue

            if node.is_leaf():
                text = (
                    f'Cluster: {node.id}\n'
                    f'P: {np.round(node.p, 3)}\n'
                    f'N: {node.sample_size}\n'
                    f'Size: {node.item_size}'
                )
                fill_colour = 'beige'

            else:
                if node.split.variable_type == 'numerical':
                    feature_name = self._numerical_feature_names[node.split.variable_idx]
                elif node.split.variable_type == 'categorical':
                    feature_name = self._categorical_feature_names[node.split.variable_idx]
                else:
                    raise RuntimeError()

                text = (
                    f'Node: {node.id}\n'
                    f'Improvement: {np.round(node.split_improvement, 2)}\n'
                    f'Feature: {feature_name}\n'
                    f'Split value: {node.split.split_value}\n'
                    f'P: {np.round(node.p, 3)}\n'
                    f'N: {node.sample_size}\n'
                    f'Size: {node.item_size}'
                )
                fill_colour = 'lightblue'

            tree_viz.node(str(node.id), text, shape='box', fillcolor=fill_colour, style='filled')

        for node in tree._nodes:
            if node.is_path_node():
                tree_viz.edge(
                    str(node.id),
                    str(node.left_child.id)
                )

                tree_viz.edge(
                    str(node.id),
                    str(node.right_child.id)
                )

        tree_viz.render(directory=output_folder, view=view)


tree = Tree(
    minimal_improvement=15,
    max_leaves=50,
    min_cluster_size=16,
    numerical_proposition_count=100,
    refine_candidates=50,
    max_items_per_node=5000,
    categorical_feature_names=['text'],
    numerical_feature_names=['X_dim', 'Y_dim', 'Z_dim', 'area',
                             'min_dim', 'max_dim', 'dim_ratio', 'weight']
)
labels = tree.fit(numerical, categorical, n, s)


clusters = np.unique(labels)
cluster_masks = []
cluster_lkhd = []

for cluster in clusters:
    cluster_masks.append(labels == cluster)
    cluster_lkhd.append(tree._nodes_dict[cluster].lkhd)

cluster_count = len(cluster_masks)

while cluster_count > 16:
    degradation_matrix = np.full(shape=(cluster_count, cluster_count), fill_value=-np.inf)
    for i in range(cluster_count):
        for j in range(cluster_count):
            if i >= j:
                continue

            mask = (cluster_masks[i] | cluster_masks[j])
            joint_lkhd = _batch_log_lkhd(n[mask], s[mask])
            lkhd_degradation = joint_lkhd - cluster_lkhd[i] - cluster_lkhd[j]

            degradation_matrix[i, j] = lkhd_degradation

    i, j = divmod(np.argmax(degradation_matrix), cluster_count)

    cluster_masks.append((cluster_masks[i] | cluster_masks[j]))
    cluster_lkhd.append(degradation_matrix[i, j] + cluster_lkhd[i] + cluster_lkhd[j])

    del cluster_masks[j]
    del cluster_masks[i]
    del cluster_lkhd[j]
    del cluster_lkhd[i]

    cluster_count -= 1


grouped_labels = np.empty_like(labels)
for i, mask in enumerate(cluster_masks):
    grouped_labels[mask] = i


m = defaultdict(set)
for l, g in zip(labels, grouped_labels):
    m[g].add(l)



evaluator = get_evaluator()


raise


skus = pd.read_csv('skus.csv')


lkhd = evaluator.evaluate(
    skus=skus.wms_sku_id.values,
    cluster_idx=grouped_labels,
)
tree.visualize_tree(view=True)

evaluator.overview(
    output_path=Path('results/lkhd_tree_pregrouped'),
    skus=skus.wms_sku_id.values,
    cluster_idx=labels,
    image_path=Path('small_images'),
    features=np.hstack([numerical, categorical]),
    feature_names=['X_dim', 'Y_dim', 'Z_dim', 'area',
                   'min_dim', 'max_dim', 'dim_ratio', 'weight', 'text']
)
tree.visualize_tree(Path('results/lkhd_tree_2'), view=True)
