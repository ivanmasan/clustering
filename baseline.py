from pathlib import Path

import numpy as np
import pandas as pd

from db import get_query_fetcher
from eval import get_evaluator, create_cluster_images
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words


splits = 4
evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

weight = skus['weight_g'].values
smallest_dim = np.stack(skus['dimensions_mm'].values).min(1)

weight = np.digitize(weight, np.quantile(weight, np.linspace(0, 1, splits + 1)[1:-1]))
smallest_dim = np.digitize(smallest_dim, np.quantile(smallest_dim, np.linspace(0, 1, splits + 1)[1:-1]))

clusters = weight * splits + smallest_dim

bic = evaluator.evaluate(
    skus=skus['wms_sku_id'].values,
    cluster_idx=clusters,
    metric='bic')

print(bic)

names = evaluator.names(
    skus=skus['wms_sku_id'].values,
    cluster_idx=clusters,
    verbose=1
)

create_cluster_images(
    skus=skus.wms_sku_id.values,
    cluster_idx=clusters,
    output_path=Path('clusters/naive_split'),
    summary_image=True,
)
