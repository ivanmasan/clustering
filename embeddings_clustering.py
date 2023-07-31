from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from evaluator.eval import create_cluster_summary_images


skus = pd.read_csv('query_data_2/skus.csv')
sku_id = skus.wms_sku_id.values

embedded_names = np.load('text_clusters/embedded_names.npz')['arr_0']

norm_emb = normalize(embedded_names)

pca = PCA(200)
norm_emb = pca.fit_transform(norm_emb)

tsne = TSNE()
tsne_emb = tsne.fit_transform(norm_emb)

np.savez(
    'text_clusters/cluster_tsne_2.npz',
    emb=tsne_emb, skus=skus.wms_sku_id.values
)

tsne_3 = TSNE(3)
tsne_emb_3 = tsne_3.fit_transform(norm_emb)

np.savez(
    'text_clusters/cluster_tsne_3.npz',
    emb=tsne_emb_3, skus=skus.wms_sku_id.values
)


clustering = KMeans(7)
cluster_vals = clustering.fit_transform(tsne_emb)

plt.scatter(*norm_emb.T, c=clustering.labels_)


lkhd = evaluator.evaluate(
    skus=sku_id,
    cluster_idx=clustering.labels_
)

evaluator.names(
    skus=sku_id,
    cluster_idx=clustering.labels_,
    verbose=True
)


create_cluster_summary_images(
    skus=sku_id,
    cluster_idx=clustering.labels_,
    output_path=Path('results/embedding_clusters/clusters_15'),
)

np.savez(
    'text_clusters/cluster_7.npz', distances=cluster_vals,
    labels=clustering.labels_, skus=skus.wms_sku_id.values
)
