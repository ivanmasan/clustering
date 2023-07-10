import matplotlib
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

matplotlib.use("Agg")

from pathlib import Path
from clearml import Task, Logger, Dataset
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from evaluator.eval import Evaluator

from scikit_utils import PartialTransformer, IQRClipper, LogarithmScaler
from autograd import AutogradClustering
import click


def _create_features(dataset_folder, include_text_features):
    feats = np.load(dataset_folder / 'X.npz', allow_pickle=True)

    X = feats['data'][:, 3:]
    feature_names = list(feats['feature_names'][3:])

    if include_text_features:
        text_cluster_path = Dataset.get(
            dataset_project='clustering',
            dataset_name='sku_text_features',
            alias='sku_text_features',
            overridable=True).get_local_copy()
        text_cluster_path = Path(text_cluster_path) / 'cluster.npz'

        text_emb = np.load(text_cluster_path.as_posix())
        X = np.hstack([X, text_emb['emb']])
        feature_names.extend([f'text_cluster_{k}' for k in range(text_emb['emb'].shape[1])])

    X = np.hstack([X, np.product(X[:, 1:3], axis=1, keepdims=True)])
    feature_names.append('exposed_surface')

    return X, feature_names


def _create_targets(dataset_folder):
    y = np.load(dataset_folder / 'weekly_failure_data.npz', allow_pickle=True)['data']

    train_y = y[:, :, :-2]
    valid_y = y[:, :, -2:]

    valid_y = valid_y.sum(2)

    return train_y, valid_y


def _get_sku_list(dataset_folder):
    data = np.load(dataset_folder / 'weekly_failure_data.npz', allow_pickle=True)
    return data['skus'].astype(int)


def _create_transformer(iqr_clipping, logarithm_transform):
    pipeline = [('scaler', StandardScaler())]

    if iqr_clipping > 0:
        iqr_clipper = IQRClipper(iqr_clipping)
        pipeline = [('iqr_clipper', iqr_clipper)] + pipeline

    if logarithm_transform:
        log_scaler = PartialTransformer(LogarithmScaler(), [3, 4, 5])
        pipeline = [('log_scaler', log_scaler)] + pipeline

    return Pipeline(pipeline)


def _apply_tsne(tsne, X):
    tsne_transformer = TSNE(tsne)
    X = tsne_transformer.fit_transform(X)
    feature_names = [f'tsne_{i}' for i in range(tsne)]
    return X, feature_names


def _eval(clustering, evaluator, task, skus, features, feature_names, episode):
    logger = Logger.current_logger()
    output_path = Path('results/temp')
    clusters = clustering.clusters(min_size=16)

    image_path = Dataset.get(
        dataset_project='clustering',
        dataset_name='sku_images',
        alias='sku_images',
        overridable=True).get_local_copy()

    lkhd = evaluator.overview(
        output_path=output_path,
        image_path=Path(image_path),
        skus=skus,
        cluster_idx=clusters,
        features=features,
        feature_names=feature_names
    )
    logger.report_single_value("Final Evaluation", lkhd)

    importances = clustering.feature_importances()
    logger = Logger.current_logger()

    logger.report_table(
        "Importances", "",
        iteration=episode,
        table_plot=importances
    )

    similarity = clustering.similarities().detach().numpy()
    similarity_path = output_path / 'similarities.npz'
    np.savez(similarity_path, sim=similarity)
    task.upload_artifact(f'Similarities {episode}', artifact_object=similarity_path)

    norm_similarity = similarity / similarity.sum(1, keepdims=True)
    most_similiar_clusters = pd.DataFrame({'sku': skus})
    most_similiar_clusters[['b5', 'b4', 'b3', 'b2', 'b1']] = np.argsort(norm_similarity, axis=1)[:, -5::]
    most_similiar_clusters[['s5', 's4', 's3', 's2', 's1']] = np.sort(norm_similarity, axis=1)[:, -5::]
    most_similiar_clusters['highest_sim'] = similarity.max(1)
    logger.report_table("Similar clusters", "", table_plot=most_similiar_clusters, iteration=episode)

    for cluster_folder in output_path.iterdir():
        if cluster_folder.is_dir():
            for image in cluster_folder.iterdir():
                if image.suffix != '.jpg':
                    continue
                logger.report_image(
                    cluster_folder.stem,
                    image.stem,
                    iteration=episode,
                    local_path=image.as_posix()
                )
        elif cluster_folder.suffix == '.jpeg':
            logger.report_image(
                "Summary Images",
                cluster_folder.stem,
                iteration=episode,
                local_path=cluster_folder.as_posix()
            )

    distance = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(2))
    same_cluster = clusters[:, None] == clusters[None, :]
    idx_x, idx_y = np.where((distance < 0.1) & (~same_cluster))
    logger.report_scalar(
        "Hard Border Skus", "", iteration=episode, value=len(idx_x)
    )

    close = pd.DataFrame({
        'Sku1': skus[idx_x], 'Sku2': skus[idx_y], 'Distance': distance[idx_x, idx_y]
    })
    logger.report_table("Close Skus", "", table_plot=close, iteration=episode)


@click.command()
@click.option('--entropy_reg_ratio', default=0.3)
@click.option('--cluster_entropy_reg', default=5)
@click.option('--time_shift_std', default=0.4)
@click.option('--within_cluster_std', default=0.5)
@click.option('--distance_decay', default=40)
@click.option('--central_cluster_reg', default=100)
@click.option('--clusters', default=48)
@click.option('--iqr_clipping', default=-1)
@click.option('--l2_reg', default=0.2)
@click.option('--include_text_features', is_flag=True, default=False)
@click.option('--logarithm_transform', is_flag=True, default=False)
@click.option('--tsne', default=-1, type=int)
@click.option('--flat_scale', is_flag=True, default=False)
def main(
    entropy_reg_ratio,
    cluster_entropy_reg,
    time_shift_std,
    within_cluster_std,
    distance_decay,
    central_cluster_reg,
    clusters,
    l2_reg,
    include_text_features,
    iqr_clipping,
    logarithm_transform,
    tsne,
    flat_scale
):
    task = Task.init(project_name="clustering", task_name="Run")
    logger = task.get_logger()

    dataset = Dataset.get(dataset_project='clustering',
                          dataset_name='failures',
                          alias='failure_data')
    dataset_folder = Path(dataset.get_local_copy())

    pipeline = _create_transformer(iqr_clipping, logarithm_transform)
    X, feature_names = _create_features(dataset_folder, include_text_features)
    if tsne > 0:
        X = pipeline.fit_transform(X)
        X, feature_names = _apply_tsne(tsne, X)
        pipeline = StandardScaler()

    train_y, valid_y = _create_targets(dataset_folder)
    skus = _get_sku_list(dataset_folder)

    evaluator = Evaluator(
        targets=valid_y,
        skus=skus,
        sku_meta_data=pd.read_csv(dataset_folder / 'skus.csv')
    )

    clustering = AutogradClustering(
        distance_decay=distance_decay,
        within_cluster_std=within_cluster_std,
        time_shift_std=time_shift_std,
        entropy_reg_ratio=entropy_reg_ratio,
        cluster_entropy_reg=cluster_entropy_reg,
        central_cluster_reg=central_cluster_reg,
        clusters=clusters,
        l2_reg=l2_reg,
        feature_names=feature_names,
        transformer=pipeline,
        X=X,
        y=train_y,
        evaluator=evaluator,
        sku_list=skus,
        flat_scale=flat_scale
    )

    for i in range(6):
        clustering.train(128, 1000)

        _eval(
            clustering=clustering,
            evaluator=evaluator,
            task=task,
            skus=skus,
            features=X,
            feature_names=feature_names,
            episode=i * 1000
        )

    task.close()


if __name__ == '__main__':
    main()
