from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from itertools import combinations

from evaluator.eval import get_evaluator


data = np.load("tree_data.npz", allow_pickle=True)
s = data.get('s')
n = data.get('n')
numerical = data.get('numerical')
categorical = data.get('categorical')

min_cluster_size = 16
numerical_proposition_count = 150


@dataclass
class Split:
    variable_type: Literal['numerical', 'categorical']
    variable_idx: int
    split_value: Union[float, set]
    mask: np.ndarray

    def __repr__(self):
        return f'{self.variable_type} idx={self.variable_idx}, value={self.split_value})'


@dataclass
class Node:
    id: float
    lkhd: float
    split: Optional[Split] = None
    split_improvement: Optional[float] = None
    left_child: Optional[Node] = None
    right_child: Optional[Node] = None
    parent: Optional[Node] = None
    parent_way: Optional[Literal['right', 'left']] = None

    def __repr__(self):
        return f'Node(id={self.id}, split_improvement={self.split_improvement}, split=({self.split}))'

    def is_leaf(self) -> bool:
        return self.split is None

    def is_orphan(self) -> bool:
        return self.split is None and self.parent is None


def _batch_log_lkhd(n, s):
    p = s.sum(0, keepdims=True) / n.sum(0, keepdims=True)
    return stats.binom(n=n, p=p).logpmf(s).sum()


def _yield_numerical_split_propositions(x):
    sort_idx = np.argsort(x)

    if len(x) < min_cluster_size * 2:
        return np.array([])

    split_idx_values = np.linspace(
        min_cluster_size,
        len(x) - min_cluster_size,
        numerical_proposition_count
    ).astype(int)

    for split_idx in split_idx_values:
        mask = np.full(shape=x.shape, fill_value=False)
        mask[sort_idx[:split_idx]] = True
        yield split_idx, mask


def _yield_categorical_split_propositions(x):
    group_counter = Counter(x)
    masks = np.zeros((x.size, x.max() + 1))
    masks[np.arange(x.size), x] = 1

    categories = set(group_counter.keys())
    total_items = sum(group_counter.values())

    for i in range(np.floor(len(categories) / 2).astype(int)):
        for subset in combinations(categories, i + 1):
            subset = set(subset)

            subset_items = sum([group_counter[s] for s in subset])
            if min_cluster_size <= subset_items <= total_items - min_cluster_size:
                yield (
                    subset,
                    sum([masks[:, s] for s in subset]).astype(bool)
                )


def _yield_all_splits(numerical, categorical):
    for i in range(numerical.shape[1]):
        generator = _yield_numerical_split_propositions(numerical[:, i])

        for split_value, mask in generator:
            yield Split(
                variable_type='numerical',
                variable_idx=i,
                split_value=split_value,
                mask=mask
            )

    for i in range(categorical.shape[1]):
        generator = _yield_categorical_split_propositions(categorical[:, i])

        for set_value, mask in generator:
            yield Split(
                variable_type='categorical',
                variable_idx=i,
                split_value=set_value,
                mask=mask
            )


def _find_best_split(numerical, categorical, n, s):
    best_split = None
    best_lkhd = -np.inf
    best_lkhd_tuple = (None, None)

    for split in _yield_all_splits(numerical, categorical):
        log_lkhd = (
            _batch_log_lkhd(n[split.mask], s[split.mask]),
            _batch_log_lkhd(n[~split.mask], s[~split.mask])
        )
        sum_log_lkhd = sum(log_lkhd)

        if sum_log_lkhd > best_lkhd:
            best_lkhd = sum_log_lkhd
            best_split = split
            best_lkhd_tuple = log_lkhd

    return best_split, best_lkhd_tuple


class Tree:
    def __init__(self, minimal_improvement=15, max_leaves=16):
        self._nodes = []
        self._nodes_dict = {}
        self._node_count = 0
        self._max_leaves = max_leaves
        self._min_split_improvement = minimal_improvement

    def _new_node(self, lkhd):
        node = Node(self._node_count, lkhd)
        self._nodes.append(node)
        self._nodes_dict[self._node_count] = node
        self._node_count += 1
        return node

    def fit(self, numerical, categorical, n, s):
        self._data_points = n.shape[0]
        self._root = self._new_node(_batch_log_lkhd(n, s))
        self._assign_children(self._root, numerical, categorical, n, s)
        self._prune()
        return self._labels()

    def _assign_children(self, node, numerical, categorical, n, s):
        if numerical.shape[0] <= min_cluster_size * 2:
            return

        split, (left_lkhd, right_lkhd) = _find_best_split(numerical, categorical, n, s)
        if split is None:
            return

        split_improvement = left_lkhd + right_lkhd - node.lkhd
        if split_improvement < self._min_split_improvement:
            return

        if split is None:
            raise

        node.split = split
        node.left_child = self._new_node(left_lkhd)
        node.left_child.parent = node
        node.left_child.parent_way = 'left'

        node.right_child = self._new_node(right_lkhd)
        node.right_child.parent = node
        node.right_child.parent_way = 'right'

        node.split_improvement = split_improvement

        self._assign_children(node.left_child, numerical[split.mask], categorical[split.mask], n[split.mask], s[split.mask])
        self._assign_children(node.right_child, numerical[~split.mask], categorical[~split.mask], n[~split.mask], s[~split.mask])

    def _leaf_nodes(self):
        for node in self._nodes:
            if node.split is not None:
                continue
            if node.parent is not None:
                yield node

    def _prune(self):
        is_removable_node = np.full(shape=self._node_count, fill_value=False)
        improvement = np.empty(shape=self._node_count)

        for node in self._nodes:
            improvement[node.id] = node.split_improvement

            if (node.split is not None
                    and node.left_child.split is None
                    and node.right_child.split is None):
                is_removable_node[node.id] = True

        leaf_count = len(list(self._leaf_nodes()))

        while leaf_count > self._max_leaves:
            idx = np.argmin(improvement[is_removable_node])
            node_id = np.where(is_removable_node)[0][idx]

            is_removable_node[node_id] = False
            node = self._nodes_dict[node_id]

            node.left_child.parent = None
            node.right_child.parent = None

            if node.parent_way == 'left':
                if node.parent.right_child.split is None:
                    is_removable_node[node.parent.id] = True
            elif node.parent_way == 'right':
                if node.parent.left_child.split is None:
                    is_removable_node[node.parent.id] = True

            node.left_child = None
            node.right_child = None
            node.split_improvement = None
            node.split = None

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


tree = Tree(minimal_improvement=15, max_leaves=32)
labels = tree.fit(numerical, categorical, n, s)


evaluator = get_evaluator()


skus = pd.read_csv('skus.csv')


lkhd = evaluator.evaluate(
    skus=skus.wms_sku_id.values,
    cluster_idx=labels,
)
print(lkhd)

if False:
    evaluator.overview(
        output_path=Path('results/custom_tree'),
        image_path=Path('images'),
        skus=skus.wms_sku_id.values,
        cluster_idx=labels,
        features=numerical,
        feature_names=['X_dim', 'Y_dim', 'Z_dim', 'area',
                       'min_dim', 'max_dim', 'dim_ratio', 'weight']
    )

numericl_feature_names = ['X_dim', 'Y_dim', 'Z_dim', 'area',
                          'min_dim', 'max_dim', 'dim_ratio', 'weight']
categorical_feature_names = ['text']


import graphviz
tree_viz = graphviz.Digraph('tree', comment='TREE')


for node in tree._nodes:
    if node.is_orphan():
        continue

    if node.is_leaf():
        mask = node.id == labels
        p = s[mask].sum((0, 2)) / n[mask].sum((0, 2))
        text = (
            f'Cluster: {node.id}\n'
            f'P: {np.round(p, 3)}\n'
            f'Size: {mask.sum()}'
        )
        fill_colour = 'beige'

    else:
        if node.split.variable_type == 'numerical':
            feature_name = numericl_feature_names[node.split.variable_idx]
        elif node.split.variable_type == 'categorical':
            feature_name = categorical_feature_names[node.split.variable_idx]

        text = (
            f'Node: {node.id}\n'
            f'Improvement: {np.round(node.split_improvement, 2)}\n'
            f'Feature: {feature_name}\n'
            f'Split value: {node.split.split_value}'
        )
        fill_colour = 'lightblue'

    tree_viz.node(str(node.id), text, shape='box', fillcolor=fill_colour, style='filled')

for node in tree._nodes:
    if not node.is_leaf():
        tree_viz.edge(
            str(node.id),
            str(node.left_child.id)
        )

        tree_viz.edge(
            str(node.id),
            str(node.right_child.id)
        )

tree_viz.render(directory='results/custom_cluster', view=True)


