import numpy as np
from sklearn.base import TransformerMixin


class PartialTransformer(TransformerMixin):
    def __init__(self, transformer, column_idx):
        self._columns = column_idx
        self._transformer = transformer

    def fit(self, X, y=None):
        filtered = X[:, self._columns]
        self._transformer.fit(filtered)
        return self

    def transform(self, X, y=None):
        ret = X.copy()
        filtered = ret[:, self._columns]
        ret[:, self._columns] = self._transformer.transform(filtered)
        return ret

    def inverse_transform(self, X, y=None):
        ret = X.copy()
        filtered = ret[:, self._columns]
        ret[:, self._columns] = self._transformer.inverse_transform(filtered)
        return ret

    def get_params(self, deep=False):
        return {
            'transformer': self._transformer,
            'column_idx': self._columns
        }


class LogarithmScaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X + 1)

    def inverse_transform(self, X, y=None):
        return np.exp(X) - 1

    def get_params(self, deep=False):
        return {}


class IQRClipper(TransformerMixin):
    def __init__(self, delta=3):
        self._delta = delta

    def fit(self, X, y=None):
        l, u = np.percentile(X, [25, 75], axis=0)
        self._lower = l - self._delta * (u - l)
        self._upper = u + self._delta * (u - l)
        return self

    def transform(self, X, y=None):
        return np.clip(X, a_min=self._lower, a_max=self._upper)

    def inverse_transform(self, X, y=None):
        return X

    def get_params(self, deep=False):
        return {'delta': self._delta}
