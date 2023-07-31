import numpy as np

from database.db import get_query_fetcher

query_fetcher = get_query_fetcher()

start_date = "2023-01-02"
end_date = "2023-07-31"

failure_data = query_fetcher.fetch(
    'FAILURE_DATA_DAILY', start_date=start_date, end_date=end_date)

sku_data = query_fetcher.fetch('SKUS')
sku_data.to_csv('query_data_2/raw_skus.csv')
sku_data = sku_data.sort_values('wms_sku_id')

failure_data = failure_data[np.isin(failure_data.sku_id.values, sku_data.wms_sku_id.values)]

skus = np.unique(failure_data.sku_id.values)
hosts = np.unique(failure_data.picker_host.values)
days = np.unique(failure_data.date.values)

data = np.zeros((len(skus), len(hosts), len(days), 5, 2))

sku_idx = np.searchsorted(skus, failure_data.sku_id)
host_idx = np.searchsorted(hosts, failure_data.picker_host)
week_idx = np.searchsorted(days, failure_data.date)

data[sku_idx, host_idx, week_idx, 0, 0] = failure_data.placefails
data[sku_idx, host_idx, week_idx, 0, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 1, 0] = failure_data.nopickattempts
data[sku_idx, host_idx, week_idx, 1, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 2, 0] = failure_data.detfails
data[sku_idx, host_idx, week_idx, 2, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 3, 0] = failure_data.error3c
data[sku_idx, host_idx, week_idx, 3, 1] = failure_data.tasks

data[sku_idx, host_idx, week_idx, 4, 0] = failure_data.fails
data[sku_idx, host_idx, week_idx, 4, 1] = failure_data.tasks

np.savez('query_data_2/daily_failure_data.npz',
         data=data, skus=skus, hosts=hosts, days=days)

