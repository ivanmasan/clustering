import os
import shutil
from pathlib import Path

from PIL import Image
from typing import Optional, Dict
from collections import defaultdict
from beartype.typing import Any, Union, List, Literal

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
        success: Union[List, np.ndarray],
        total: Union[List, np.ndarray],
        groups: Optional[Union[List, np.ndarray]] = None,
        sku_meta_data: pd.DataFrame = None,
    ) -> None:
        if groups is None:
            self._groups = np.zeros(len(skus))
        else:
            groups = _to_numpy_array(groups, to_2d=True)
            self._groups = self._generate_group_idx(groups)

        self._skus = _to_numpy_array(skus)
        self._success = _to_numpy_array(success, to_2d=True)
        self._total = _to_numpy_array(total, to_2d=True)

        self._sku_id_to_name = {}
        if sku_meta_data is not None:
            for _, row in sku_meta_data.iterrows():
                self._sku_id_to_name[row['wms_sku_id']] = row['name']

        self._precompute_combination_numbers()

    def _generate_group_idx(self, groups: np.ndarray):
        group_sort_idx = np.lexsort(groups.T)
        ordered_groups = groups[group_sort_idx]
        group_start_idx = np.where(
            np.any(ordered_groups[1:] != ordered_groups[:-1], axis=1)
        )[0]

        assigned_group = np.searchsorted(group_start_idx, np.arange(len(groups)))

        reverse_group_sort_idx = np.argsort(group_sort_idx)
        return assigned_group[reverse_group_sort_idx]

    def _precompute_combination_numbers(self):
        self._log_comb_lkhd = (
            gammaln(self._total + 1)
            - gammaln(self._success + 1)
            - gammaln(self._total - self._success + 1)
        )

    @beartype
    def evaluate(
        self,
        skus: Union[List, np.ndarray],
        cluster_idx: Union[List, np.ndarray],
        metric: Literal["log_lkhd", "loo_log_lkhd", "aic", "bic"] = 'log_lkhd',
    ) -> float:
        skus = _to_numpy_array(skus)
        cluster_idx = _to_numpy_array(cluster_idx)

        group_idxs = self._cluster_idx(skus, cluster_idx)

        use_loo_lkhd = metric == "loo_log_lkhd"

        total_log_lkhd = 0
        for cluster_idx in group_idxs:
            if np.isnan(self._compute_log_lkhd(cluster_idx, loo=use_loo_lkhd)):
                break
            total_log_lkhd += self._compute_log_lkhd(cluster_idx, loo=use_loo_lkhd)

        k = len(group_idxs) * self._success.shape[1]
        n = self._total.sum()

        if metric == "aic":
            return 2 * k - 2 * total_log_lkhd
        elif metric == "bic":
            return np.log(n) * k - 2 * total_log_lkhd
        else:
            return total_log_lkhd

    def _cluster_idx(self, skus: np.ndarray, cluster_idx: np.ndarray) -> List[np.ndarray]:
        cluster_dict = {}

        for sku, cluster_id in zip(skus, cluster_idx):
            cluster_dict[sku] = cluster_id

        cluster_idx_mapping = defaultdict(list)

        for i, (sku, group) in enumerate(zip(self._skus, self._groups)):
            try:
                cluster_id = cluster_dict[sku]
            except KeyError:
                raise ValueError(f"Sku {sku} is missing from cluster specification")

            cluster_idx_mapping[(cluster_id, group)].append(i)

        return [np.array(x) for x in cluster_idx_mapping.values()]

    def _compute_log_lkhd(self, cluster_idx: np.ndarray, loo: bool) -> float:
        total = self._total[cluster_idx]
        success = self._success[cluster_idx]
        log_comb_lkhd = self._log_comb_lkhd[cluster_idx]

        sum_s = success.sum(0)
        sum_n = total.sum(0)

        if loo:
            p = (sum_s - success) / (sum_n - total)
        else:
            p = sum_s / sum_n

        p[np.isnan(p)] = 0

        return (
            log_comb_lkhd
            + np.log(np.clip(p, 1e-10, 1)) * success
            + np.log(np.clip(1 - p, 1e-10, 1)) * (total - success)
        ).sum()

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
    ) -> None:
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


def prepare_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)


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


def get_evaluator():
    eval_data = pd.read_csv('old_data/eval_data.csv')
    eval_data = eval_data.fillna(0)

    return Evaluator(
        groups=eval_data[['week', 'picker_host']].values,
        skus=eval_data['wms_sku_id'].values,
        success=eval_data[['task_success', 'pick_success', 'place_success']].values,
        total=eval_data[['task_total', 'pick_total', 'place_total']].values,
        sku_meta_data=pd.read_csv('old_data/skus.csv')
    )
