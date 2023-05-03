import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from evaluator.eval import get_evaluator
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from unidecode import unidecode
from torch import nn

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

evaluator = get_evaluator()


skus = pd.read_csv('skus.csv')
skus['dimensions_mm'] = skus['dimensions_mm'].apply(lambda x: [int(xx) for xx in x.strip('][').split(', ')])


vect = CountVectorizer(
    strip_accents='unicode',
    stop_words=[unidecode(x) for x in get_stop_words('czech')],
    max_features=150
)

bow_feats = vect.fit_transform(skus['name'])

sku_features = np.concatenate([
        np.stack(skus['dimensions_mm'].values),
        skus[['weight_g']].values,
        bow_feats.A], axis=1)


train_data = pd.read_csv('train_data_week.csv')
train_data = train_data.fillna(0)

success_y = torch.Tensor(train_data[['task_success', 'pick_success', 'place_success']].values)
total_y = torch.Tensor(train_data[['task_total', 'pick_total', 'place_total']].values)
failure_y = total_y - success_y


sku_to_train_idx = np.searchsorted(skus['wms_sku_id'], train_data['wms_sku_id'])
X = sku_features[sku_to_train_idx]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.Tensor(X)

aux_X = train_data[['picker_host', 'week']].values

ohe = OneHotEncoder(sparse=False, drop='first')
aux_scaler = StandardScaler()

aux_X = ohe.fit_transform(aux_X)
aux_X = aux_scaler.fit_transform(aux_X)
aux_X = torch.Tensor(aux_X)


embedder = nn.Sequential(
    nn.LazyLinear(128),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.LazyLinear(64),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.LazyLinear(32),
).to("cuda")


class Concatenate(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, *args):
        feats = torch.cat(args, dim=1)
        return self._model(feats)


rate_predictor = Concatenate(
    nn.Sequential(
        nn.LazyLinear(64),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.LazyLinear(3),
        nn.Sigmoid()
    )
).to("cuda")


reconstructor = nn.Sequential(
    nn.LazyLinear(64),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.LazyLinear(128),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.LazyLinear(X.shape[1]),
).to("cuda")


optimizer = torch.optim.Adam(
    params=(list(rate_predictor.parameters())
            + list(embedder.parameters())
            + list(reconstructor.parameters()))
)
reconstruction_loss = torch.nn.MSELoss()


def yield_batches(size: int, batch_size: int, episodes: int = 1):
    for _ in range(episodes):
        idx = np.arange(size)
        np.random.shuffle(idx)

        while len(idx):
            yield idx[:batch_size]
            idx = idx[batch_size:]


for i in range(10):
    for idx in yield_batches(X.shape[0], 128):
        features = X[idx].to("cuda")
        embeddings = embedder(features)
        y_pred = rate_predictor(embeddings, aux_X[idx].to("cuda"))
        reconstructed_features = reconstructor(embeddings)

        y_pred = torch.clip(y_pred, 1e-6, 1 - 1e-6)
        lkhd = - (torch.log(y_pred) * success_y[idx].to("cuda")
                  + torch.log(1 - y_pred) * failure_y[idx].to("cuda")).sum()

        loss = reconstruction_loss(reconstructed_features, features)

        total_loss = 5 * loss + lkhd

        optimizer.zero_grad()
        lkhd.backward()
        optimizer.step()

    embeddings = embedder(torch.Tensor(sku_features).to("cuda"))
    embeddings = embeddings.to("cpu").detach().numpy()
    embeddings = normalize(embeddings, axis=1)

    dim_red = TSNE()
    new_dim = dim_red.fit_transform(embeddings)

    clustering = KMeans(n_clusters=16)
    clustering.fit_transform(new_dim)

    clusters = []
    for l in range(max(clustering.labels_) + 1):
        mask = clustering.labels_ == l
        clusters.append(skus['wms_sku_id'].values[mask])

    print(evaluator.evaluate(clusters, metric='bic'))
