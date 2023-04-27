import pandas as pd
import numpy as np
from scipy.special import gammaln

data = pd.read_csv('data.csv')

data = data[~data['pick_success'].isna()]

data = data[data.picker_host == 'ropi01']

command_id = data.command_id.values
first_idx = np.where((command_id[:-1] != command_id[1:]))[0] + 1
first_idx = np.insert(first_idx, 0, 0)

data = data.iloc[first_idx]


data = data.groupby(['sku_name']).agg({
    'pick_success': ['sum', 'count'],
    'place_success': ['sum', 'count'],
    'sku_min_dim_mm': 'min',
    'sku_weight_g': 'min'
})


def log_lkhd(cluster_data):
    s = cluster_data['pick_success']['sum'].values.astype(int)
    n = cluster_data['pick_success']['count'].values.astype(int)

    p = s.sum() / n.sum()

    if p == 0 or p == 1:
        return 0

    return (
        s * np.log(p)
        + (n - s) * np.log(1 - p)
        + gammaln(n + 1)
        - gammaln(s + 1)
        - gammaln(n - s + 1)
    ).sum()


def loo_log_lkhd(cluster_data):
    s = cluster_data['pick_success']['sum'].values.astype(int)
    n = cluster_data['pick_success']['count'].values.astype(int)

    n_sum = n.sum()
    s_sum = s.sum()

    p = (s_sum - s) / (n_sum - n)

    return (
        s * np.log(p)
        + (n - s) * np.log(1 - p)
        + gammaln(n + 1)
        - gammaln(s + 1)
        - gammaln(n - s + 1)
    ).sum()


values = data['sku_min_dim_mm'].values
split_values = np.quantile(values, np.linspace(0, 1, 10)[1:-1])
bins = np.digitize(values, split_values)
data['cluster'] = bins

data['cluster'] = np.random.randint(10, size=data.shape[0])


total = 0
total_loo = 0

for _, cluster_data in data.groupby(['cluster']):
    total += log_lkhd(cluster_data)
    total_loo += loo_log_lkhd(cluster_data)

k = data['cluster'].nunique()
n = data['pick_success']['count'].sum()

print('AIC:', 2 * k - 2 * total)
print('BIC:', np.log(n) * k - 2 * total)
print('likehood:', total)
print('loo_likehood:', total_loo)
