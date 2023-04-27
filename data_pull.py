from db import get_query_fetcher
import pandas as pd
from PIL import Image
import requests
from io import BytesIO


query_fetcher = get_query_fetcher()

eval_data = query_fetcher.fetch('DATA_BY_WEEK', start_date="'2023-02-27'", end_date="'2023-04-02'")
eval_data = eval_data.fillna(0)
eval_data.to_csv('eval_data.csv')


train_data = query_fetcher.fetch('DATA_TOTAL', start_date="'2023-01-02'", end_date="'2023-02-26'")
train_data.to_csv('train_data.csv')

train_data = query_fetcher.fetch('DATA_BY_WEEK', start_date="'2023-01-02'", end_date="'2023-02-26'")
train_data.to_csv('train_data_week.csv')


skus = query_fetcher.fetch('SKUS')
skus.to_csv('skus.csv')


skus = pd.read_csv('skus.csv')
for _, (sku, image_url) in skus.iloc[358:][['wms_sku_id', 'image_url']].iterrows():
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    if img.mode == 'CMYK':
        img = img.convert('RGB')

    extension = image_url.split('.')[-1]

    with open(f'images/{sku}.png', 'wb') as f:
        img.save(f)
