from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from evaluator.eval import get_evaluator, create_cluster_summary_images


evaluator = get_evaluator()

skus = pd.read_csv('query_data/skus.csv')
sku_id = skus.wms_sku_id.values

embedded_names = np.load('text_clusters/embedded_names.npz')['arr_0']

norm_emb = normalize(embedded_names)

pca = PCA(200)
norm_emb = pca.fit_transform(norm_emb)

tsne = TSNE()
norm_emb = tsne.fit_transform(norm_emb)

clustering = KMeans(7)
cluster_vals = clustering.fit_transform(norm_emb)

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
