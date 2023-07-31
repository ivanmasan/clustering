from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

skus = pd.read_csv('query_data_2/skus.csv')
pulled_images = [x.stem for x in Path('images').iterdir()]

for _, (sku, image_url) in tqdm(skus[['wms_sku_id', 'image_url']].iterrows()):
    if str(sku) in pulled_images:
        continue
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    if img.mode == 'CMYK':
        img = img.convert('RGB')

    extension = image_url.split('.')[-1]

    with open(f'original_images/{sku}.png', 'wb') as f:
        img.save(f)

    scale_factor = 150 / max(img.size)
    img = img.resize(size=(int(img.size[0] * scale_factor),
                           int(img.size[1] * scale_factor)))
    img = img.convert('RGB')

    with open(f'images/{sku}.jpg', 'wb') as f:
        img.save(f, "JPEG", quality=70)
