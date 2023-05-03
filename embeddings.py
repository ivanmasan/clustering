import fasttext.util
import numpy as np
import pandas as pd

fasttext.util.download_model('cs', if_exists='ignore')
ft = fasttext.load_model('cc.cs.300.bin')

skus = pd.read_csv('query_data/skus.csv')
names = skus.name.apply(lambda x: x.lower()).values

embedded_names = []
for name in names:
    embedding = ft.get_sentence_vector(name)
    embedded_names.append(embedding)

embedded_names = np.stack(embedded_names)

np.savez('text_clusters/embedded_names.npz', embedded_names)
