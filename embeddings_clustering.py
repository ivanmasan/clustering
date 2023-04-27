from pathlib import Path

import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

from eval import get_evaluator, create_cluster_images

evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
sku_id = skus.wms_sku_id.values

embedded_names = np.load('embedded_names.npz')['arr_0']

norm_emb = normalize(embedded_names)

pca = PCA(200)
norm_emb = pca.fit_transform(norm_emb)

tsne = TSNE()
norm_emb = tsne.fit_transform(norm_emb)

clustering = KMeans(8)
cluster_vals = clustering.fit_transform(norm_emb)


lkhd = evaluator.evaluate(
    skus=sku_id,
    cluster_idx=clustering.labels_,
    metric='log_lkhd'
)

evaluator.names(
    skus=sku_id,
    cluster_idx=clustering.labels_,
    verbose=True
)

create_cluster_images(
    skus=sku_id,
    cluster_idx=clustering.labels_,
    output_path=Path('embedding_clusters/clusters'),
    summary_image=True,
    image_subfolders=True
)

np.savez('text_clusters.npz', cluster_vals)
np.savez('text_clusters_labels.npz', clustering.labels_)
