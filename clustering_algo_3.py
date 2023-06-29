from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from evaluator.eval import get_evaluator

import torch


softplus = torch.nn.Softplus()


def point_similarity(clusters, scale, X):
    difference = X[:, None, :] - clusters[None, :, :]

    difference = difference * softplus(scale)[None, :, :]
    distance = (difference ** 2).sum(2)
    distance = torch.exp(-distance / 15)

    return distance


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


evaluator = get_evaluator()

skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])

sku_text_embedding = np.load('text_clusters.npz')['arr_0']


train_data = pd.read_csv('train_data.csv')
train_data = train_data.fillna(0)

s_cols = ['task_success', 'pick_success', 'place_success']
n_cols = ['task_total', 'pick_total', 'place_total']

train_data = train_data.pivot(
    index=['wms_sku_id'], columns=['picker_host'],
    values=s_cols + n_cols).fillna(0)


sku_to_train_idx = np.searchsorted(skus['wms_sku_id'], train_data.index)

s_targets = np.zeros((skus.shape[0], len(n_cols), train_data[n_cols[0]].shape[1]))
n_targets = np.zeros((skus.shape[0], len(n_cols), train_data[n_cols[0]].shape[1]))

for i, (s_col, n_col) in enumerate(zip(s_cols, n_cols)):
    s_targets[sku_to_train_idx, i] = train_data[s_col].values
    n_targets[sku_to_train_idx, i] = train_data[n_col].values

s_targets = torch.tensor(s_targets)
n_targets = torch.tensor(n_targets)


dimensions = np.stack(skus['dimensions_mm'].values)

uscaled_X = np.concatenate([
    dimensions,
    dimensions.prod(1, keepdims=True),
    dimensions.min(1, keepdims=True),
    dimensions.max(1, keepdims=True),
    dimensions.max(1, keepdims=True) / dimensions.min(1, keepdims=True),
    skus[['weight_g']].values,
    sku_text_embedding], axis=1)


#for i in range(sku_features.shape[1]):
#    sku_features[:, i] = boxcox(sku_features[:, i])[0]

sc = StandardScaler()
X = sc.fit_transform(uscaled_X)

X = torch.tensor(X)

SAMPLES = X.shape[0]
FEATURES = X.shape[1]
CLUSTERS = 12
METRICS = len(n_cols)
METRIC_SEGMENTS = n_targets[0].shape[1]

clusters = torch.Tensor(np.random.normal(0, 0.05, size=(CLUSTERS, FEATURES)))
clusters.requires_grad = True

p = torch.zeros((CLUSTERS, METRICS, METRIC_SEGMENTS), requires_grad=True)
theta = torch.zeros((CLUSTERS, METRICS), requires_grad=True)

scale = torch.zeros((CLUSTERS, FEATURES), requires_grad=True)
optimizer = torch.optim.SGD([clusters, p, scale, theta], lr=0.01)

i = 0
for batch in yield_repeat_batches(SAMPLES, episodes=5000):
    similarity = point_similarity(clusters, scale, X[batch])
    similarity = similarity / similarity.sum(1, keepdim=True)

    avg_p = (similarity[:, :, None, None] * torch.sigmoid(p)[None, :, :, :]).sum(1)

    binom = torch.distributions.binomial.Binomial(total_count=n_targets[batch], probs=avg_p)
    data_log_lkhd = binom.log_prob(s_targets[batch]).sum()

    norm = torch.distributions.normal.Normal(theta, scale=0.1)
    model_log_lkhd = norm.log_prob(p.swapaxes(1, 2).swapaxes(0, 1)).sum()

    neg_log_lkhd = - (model_log_lkhd + data_log_lkhd)

    if neg_log_lkhd.isnan():
        raise RuntimeError()

    optimizer.zero_grad()
    neg_log_lkhd.backward()
    torch.nn.utils.clip_grad_norm_([clusters, p, scale, theta], 20)
    optimizer.step()

    i += 1
    if i % 50 == 0:
        similarity = point_similarity(clusters, scale, X)
        inference_clusters = torch.argmax(similarity, dim=1).numpy()
        print(Counter(inference_clusters))

        log_lkhd = evaluator.evaluate(
            skus=skus.wms_sku_id.values,
            cluster_idx=inference_clusters,
            metric='log_lkhd'
        )

        print(log_lkhd)
        print(neg_log_lkhd)


evaluator.overview(
    output_path=Path('results/custom_cluster'),
    image_path=Path('images'),
    skus=skus.wms_sku_id.values,
    cluster_idx=clusters,
    features=uscaled_X,
    feature_names=['X_dim', 'Y_dim', 'Z_dim', 'area',
                   'min_dim', 'max_dim', 'dim_ratio', 'weight']
                  + [f'text_cluster_{i}' for i in range(8)]
)


skus_to_investigate
914016
910011
1466927
