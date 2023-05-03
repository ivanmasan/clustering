import numpy as np
import pandas as pd

from database.db import get_connector

connector = get_connector()

clusters = np.load('results/lkhd_tree_new/clusters.npz')

cluster_df = pd.DataFrame({
    'wms_sku_id': clusters['skus'].astype(str),
    'cluster': np.searchsorted(np.unique(clusters['labels']), clusters['labels']),
})


connector.push(cluster_df, 'clusters', index=False, if_exists='replace')


cluster_meta_data = pd.read_csv('results/lkhd_tree_new/cluster_summary.csv')
cluster_meta_data['cluster_id'] = np.arange(len(cluster_meta_data))
cluster_meta_data['supercluster_id'] = np.searchsorted(np.unique(cluster_meta_data['supercluster']), cluster_meta_data['supercluster'])


cluster_meta_data = cluster_meta_data.rename({
    'cluster': 'cluster_tree_node_id',
    'supercluster': 'supercluster_tree_node_id',
    'name': 'cluster_name',
    'superclustername': 'supercluster_name'
}, axis=1)

cluster_meta_data = cluster_meta_data[
    ['cluster_id', 'cluster_tree_node_id',
     'supercluster_id', 'supercluster_tree_node_id',
     'cluster_name', 'supercluster_name']
]

connector.push(cluster_meta_data, 'clusters_metadata', index=False)


