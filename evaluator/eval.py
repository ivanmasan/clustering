import os
import shutil
from pathlib import Path

from PIL import Image
from typing import Optional, Dict
from collections import defaultdict
from beartype.typing import Any, Union, List, Literal

from scipy.stats import binom
import pandas as pd
import numpy as np
from scipy.special import gammaln
from beartype import beartype
from tqdm import tqdm


class Evaluator:
    @beartype
    def __init__(
        self,
        skus: Union[List, np.ndarray],
        targets: np.ndarray,
        sku_meta_data: pd.DataFrame = None,
    ) -> None:
        self._skus = _to_numpy_array(skus)
        self._targets = targets

        assert targets.shape[0] == len(self._skus)
        assert targets.shape[3] == 2

        self._sku_to_idx = {}
        for idx, sku in enumerate(self._skus):
            self._sku_to_idx[sku] = idx

        self._sku_id_to_name = {}
        if sku_meta_data is not None:
            for _, row in sku_meta_data.iterrows():
                self._sku_id_to_name[row['wms_sku_id']] = row['name']

    @beartype
    def evaluate(
        self,
        skus: Union[List, np.ndarray],
        cluster_idx: Union[List, np.ndarray],
        per_metric: bool = False
    ) -> Union[np.ndarray, float]:
        skus = _to_numpy_array(skus)
        cluster_idx = _to_numpy_array(cluster_idx)

        cluster_dict = self._cluster_dict(skus, cluster_idx)

        log_lkhd = np.zeros(self._targets.shape[2])
        for _, cluster_items_idx in cluster_dict.items():
            cluster_skus = self._targets[cluster_items_idx]

            cluster_target = cluster_skus.sum(0)
            p = cluster_target[:, :, 0] / cluster_target[:, :, 1]
            p[np.isnan(p)] = 0

            dist = binom(n=cluster_skus[:, :, :, 1], p=p[None, :, :])
            log_lkhd += dist.logpmf(cluster_skus[:, :, :, 0]).sum((0, 1))

        if per_metric:
            return log_lkhd
        else:
            return log_lkhd.sum()

    def _cluster_dict(self, skus: np.ndarray, cluster_idx: np.ndarray) -> dict:
        cluster_dict = defaultdict(list)

        for sku, cluster_id in zip(skus, cluster_idx):
            try:
                sku_id = self._sku_to_idx[sku]
            except KeyError:
                pass
            else:
                cluster_dict[cluster_id].append(sku_id)

        return cluster_dict

    def names(
        self,
        skus: Union[List, np.ndarray],
        cluster_idx: Union[List, np.ndarray],
        verbose: bool = False,
        print_limit: int = 10
    ) -> Dict:
        skus = _to_numpy_array(skus)
        cluster_idx = _to_numpy_array(cluster_idx)

        names = defaultdict(list)
        for sku, cluster_id in zip(skus, cluster_idx):
            sku_name = self._sku_id_to_name.get(sku)
            if sku_name is not None:
                names[cluster_id].append(sku_name)

        if verbose:
            for cluster_id, cluster_items in names.items():
                print('')
                print('---CLUSTER: ', cluster_id)
                if len(cluster_items) <= print_limit:
                    items = cluster_items
                else:
                    items = np.random.choice(cluster_items, print_limit, replace=False)
                for item in items:
                    print(item)

        return names

    def create_spreadsheets(
        self,
        skus: np.ndarray,
        features: np.ndarray,
        cluster_idx: np.ndarray,
        feature_names: List[str],
        output_path: Path
    ) -> None:
        output_path.mkdir(exist_ok=True)

        data = pd.DataFrame(features, columns=feature_names)
        data['sku_id'] = skus
        data['sku_name'] = data['sku_id'].apply(self._sku_id_to_name.get)
        data['cluster'] = cluster_idx
        data.to_csv(output_path / 'sku_summary.csv')

        for cluster, segment in data.groupby('cluster'):
            feature_quantiles = segment[feature_names].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
            cluster_folder = output_path / str(cluster)
            cluster_folder.mkdir(exist_ok=True)
            feature_quantiles.to_csv(cluster_folder / 'feature_quantiles.csv')

    def overview(
        self,
        output_path: Path,
        image_path: Path,
        skus: np.ndarray,
        cluster_idx: np.ndarray,
        features: np.ndarray,
        feature_names: Union[List, np.ndarray]
    ) -> float:
        prepare_folder(output_path)

        create_cluster_summary_images(
            skus=skus,
            cluster_idx=cluster_idx,
            output_path=output_path,
            image_path_source=image_path
        )

        create_cluster_image_folders(
            skus=skus,
            cluster_idx=cluster_idx,
            output_path=output_path,
            image_path_source=image_path,
            sym_link=True
        )

        self.create_spreadsheets(
            skus=skus,
            cluster_idx=cluster_idx,
            features=features,
            feature_names=feature_names,
            output_path=output_path
        )

        lkhd = self.evaluate(
            skus=skus,
            cluster_idx=cluster_idx
        )

        print(lkhd)

        with open(output_path / 'log_lkhd.txt', 'w') as f:
            f.write(str(lkhd))

        return lkhd


def prepare_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)


def _create_skus_by_cluster(
    skus: Union[List, np.ndarray],
    cluster_idx: Union[List, np.ndarray],
) -> Dict:
    skus_by_cluster = defaultdict(list)

    for sku, cluster_id in zip(skus, cluster_idx):
        skus_by_cluster[cluster_id].append(sku)

    return skus_by_cluster


def _sample_items(items: np.ndarray, sample_count: int) -> np.ndarray:
    if len(items) <= sample_count:
        return items
    else:
        return np.random.choice(items, sample_count, replace=False)


def _sku_to_image_path_dict(image_path: Path) -> Dict[int, Path]:
    sku_to_image_path = {}

    for item in image_path.iterdir():
        if item.is_dir():
            continue
        sku_id = item.name.split('.')[0]
        sku_to_image_path[int(sku_id)] = item

    return sku_to_image_path


def create_cluster_summary_images(
    skus: Union[List, np.ndarray],
    cluster_idx: Union[List, np.ndarray],
    output_path: Path,
    images_per_cluster: int = 40,
    image_path_source: Path = None) -> None:

    if image_path_source is None:
        image_path_source = Path('images')

    skus_by_cluster = _create_skus_by_cluster(skus, cluster_idx)

    for cluster_id, cluster_skus in tqdm(skus_by_cluster.items()):
        sampled_skus = _sample_items(cluster_skus, images_per_cluster)

        cluster_image = _create_cluster_image(sampled_skus, image_path_source)
        cluster_image = Image.fromarray(cluster_image.astype(np.uint8))
        cluster_image.save(f"{output_path}/{cluster_id}.jpeg")


def create_cluster_image_folders(
    skus: Union[List, np.ndarray],
    cluster_idx: Union[List, np.ndarray],
    output_path: Path,
    images_per_cluster: int = np.inf,
    image_path_source: Path = None,
    sym_link: bool = False
) -> None:
    if image_path_source is None:
        image_path_source = Path('../small_images')
    sku_id_to_image_path = _sku_to_image_path_dict(image_path_source)

    skus_by_cluster = _create_skus_by_cluster(skus, cluster_idx)

    for cluster_id, cluster_skus in tqdm(skus_by_cluster.items()):
        sampled_skus = _sample_items(cluster_skus, images_per_cluster)

        cluster_folder = output_path / str(cluster_id)
        cluster_folder.mkdir()

        relative_image_path_source = os.path.relpath(image_path_source, cluster_folder)

        for sku_id in sampled_skus:
            sku_image_path = sku_id_to_image_path[sku_id]
            if sym_link:
                os.symlink(
                    src=f'{relative_image_path_source}/{sku_image_path.name}',
                    dst=cluster_folder / sku_image_path.name
                )
            else:
                shutil.copy(
                    sku_image_path,
                    cluster_folder / sku_image_path.name
                )


def _create_cluster_image(skus: np.ndarray, image_folder: Path) -> np.ndarray:
    sku_id_to_image_path = _sku_to_image_path_dict(image_folder)

    cluster_image = np.full((800, 800, 3), fill_value=255)
    for sku_id in skus:
        with open(sku_id_to_image_path[sku_id].as_posix(), 'rb') as f:
            im = Image.open(f)
            scale_factor = 200 / max(im.size)
            im = im.resize(size=(int(im.size[0] * scale_factor),
                                 int(im.size[1] * scale_factor)))
        x = np.random.randint(600)
        y = np.random.randint(600)
        im = np.array(im)

        if im.ndim == 2:
            continue

        elif im.shape[2] == 4:
            im = im[:, :, :3]

        mask = np.all(im != 255, axis=2)

        y_mask, x_mask = np.where(mask)
        im = im[y_mask.min():y_mask.max(), x_mask.min():x_mask.max()]

        cluster_image[y:(y + im.shape[0]), x:(x + im.shape[1])] = im
    return cluster_image


def _to_numpy_array(
    x: Any, to_2d: bool = False
) -> np.ndarray:
    if isinstance(x, np.ndarray):
        pass
    elif isinstance(x, list):
        x = np.array(x)
    else:
        raise TypeError("Unable to convert to numpy")

    if x.ndim == 1 and to_2d:
        x = x.reshape(-1, 1)

    return x
