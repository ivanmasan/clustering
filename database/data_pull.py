import numpy as np

from database.db import get_query_fetcher

query_fetcher = get_query_fetcher()

start_date = "2023-01-02"
end_date = "2023-04-27"

first_pick_data = query_fetcher.fetch(
    'DATA_WEEKLY', start_date=start_date, end_date=end_date)
failure_data = query_fetcher.fetch(
    'FAILURE_DATA_WEEKLY', start_date=start_date, end_date=end_date)

sku_data = query_fetcher.fetch('SKUS')
sku_data.to_csv('query_data/raw_skus.csv')
sku_data = sku_data.sort_values('wms_sku_id')

first_pick_data = first_pick_data[np.isin(first_pick_data.sku_id.values, sku_data.wms_sku_id.values)]
failure_data = failure_data[np.isin(failure_data.sku_id.values, sku_data.wms_sku_id.values)]


first_pick_data = first_pick_data.fillna(0)

skus = np.concatenate([first_pick_data.sku_id.values, failure_data.sku_id.values])
skus = np.unique(skus)

hosts = np.concatenate([first_pick_data.picker_host.values, failure_data.picker_host.values])
hosts = np.unique(hosts)

weeks = np.concatenate([first_pick_data.week.values, failure_data.week.values])
weeks = np.unique(weeks)

data = np.zeros((len(skus), len(hosts), len(weeks), 7, 2))

sku_idx = np.searchsorted(skus, first_pick_data.sku_id)
host_idx = np.searchsorted(hosts, first_pick_data.picker_host)
week_idx = np.searchsorted(weeks, first_pick_data.week)

data[sku_idx, host_idx, week_idx, 0, 0] = first_pick_data.pick_total - first_pick_data.pick_success
data[sku_idx, host_idx, week_idx, 0, 1] = first_pick_data.pick_total

data[sku_idx, host_idx, week_idx, 1, 0] = first_pick_data.place_total - first_pick_data.place_success
data[sku_idx, host_idx, week_idx, 1, 1] = first_pick_data.place_total

sku_idx = np.searchsorted(skus, failure_data.sku_id)
host_idx = np.searchsorted(hosts, failure_data.picker_host)
week_idx = np.searchsorted(weeks, failure_data.week)

data[sku_idx, host_idx, week_idx, 2, 0] = failure_data.placefails
data[sku_idx, host_idx, week_idx, 2, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 3, 0] = failure_data.nopickattempts
data[sku_idx, host_idx, week_idx, 3, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 4, 0] = failure_data.detfails
data[sku_idx, host_idx, week_idx, 4, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 5, 0] = failure_data.error3c
data[sku_idx, host_idx, week_idx, 5, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 6, 0] = failure_data.fails
data[sku_idx, host_idx, week_idx, 6, 1] = failure_data.tasks

np.savez('query_data/data.npz', data=data, skus=skus, hosts=hosts, weeks=weeks)


sku_idx = np.searchsorted(sku_data.wms_sku_id, skus)
sku_data = sku_data.iloc[sku_idx]
sku_data.to_csv('query_data/skus.csv')

dimensions = np.stack(sku_data['dimensions_mm'].values)

X = np.concatenate([
    dimensions,
    np.sort(dimensions, axis=1),
    dimensions.prod(1, keepdims=True),
    dimensions.max(1, keepdims=True) / dimensions.min(1, keepdims=True),
    sku_data[['weight_g']].values], axis=1
)
feature_names = ['X_dim', 'Y_dim', 'Z_dim',
                 'min_dim', 'mid_dim', 'max_dim',
                 'area', 'dim_ratio', 'weight']

np.savez('query_data/X.npz', data=X, skus=skus, feature_names=feature_names)
