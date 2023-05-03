import pytest
import numpy as np
from beartype.roar import BeartypeCallHintParamViolation
from scipy.stats import binom

from eval import Evaluator


@pytest.fixture
def base_evaluator():
    return Evaluator(
        skus=["a", "b", "c", "d"], success=[1, 3, 1, 1], total=[4, 4, 2, 3]
    )


@pytest.fixture
def meta_data_evaluator():
    return Evaluator(
        skus=["a", "b", "c", "d"],
        success=[1, 3, 1, 1],
        total=[4, 4, 2, 3],
        sku_names=["name_a", 'name_b', "name_c", "name_d"]
    )


@pytest.fixture(params=["simple", "multigroup"])
def group_evaluator(request):
    if request.param == "simple":
        return Evaluator(
            skus=["a", "a", "b", "c", "d", "d"],
            groups=["g1", "g2", "g1", "g2", "g1", "g2"],
            success=[1, 3, 1, 1, 1, 1],
            total=[4, 4, 2, 3, 2, 2],
        )
    elif request.param == "multigroup":
        return Evaluator(
            skus=["a", "a", "b", "c", "d", "d"],
            groups=np.array([
                ["g0", "g0", "g0", "g0", "g0", "g0"],
                ["g1", "g2", "g1", "g2", "g1", "g2"],
            ]).T,
            success=[1, 3, 1, 1, 1, 1],
            total=[4, 4, 2, 3, 2, 2],
        )


@pytest.fixture
def multi_success_evaluator():
    return Evaluator(
        skus=["a", "b", "c", "d"],
        success=np.array([[1, 3, 1, 1], [1, 1, 1, 1]]).T,
        total=np.array([[4, 4, 2, 3], [3, 3, 2, 2]]).T,
    )


def test_log_lkhd_multi_success_evaluation(multi_success_evaluator):
    lkhd = multi_success_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="log_lkhd"
    )
    exp_lkhd = (
        binom(n=4, p=0.5).logpmf(1)
        + binom(n=4, p=0.5).logpmf(3)
        + binom(n=3, p=0.4).logpmf(1)
        + binom(n=2, p=0.4).logpmf(1)
        + binom(n=3, p=1 / 3).logpmf(1)
        + binom(n=3, p=1 / 3).logpmf(1)
        + binom(n=2, p=0.5).logpmf(1)
        + binom(n=2, p=0.5).logpmf(1)
    )
    assert np.abs(lkhd - exp_lkhd) < 1e-6


@pytest.mark.skip
def test_loo_log_lkhd_multi_success_evaluation(base_evaluator):
    lkhd = base_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="loo_log_lkhd"
    )
    exp_lkhd = (
        binom(n=4, p=0.5).logpmf(1)
        + binom(n=4, p=0.5).logpmf(3)
        + binom(n=3, p=0.4).logpmf(1)
        + binom(n=2, p=0.4).logpmf(1)
        + binom(n=3, p=1 / 3).logpmf(1)
        + binom(n=3, p=1 / 3).logpmf(1)
        + binom(n=2, p=0.5).logpmf(1)
        + binom(n=2, p=0.5).logpmf(1)
    )
    assert np.abs(lkhd - exp_lkhd) < 1e-6


def test_log_lkhd_grouped_evaluation(group_evaluator):
    lkhd = group_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="log_lkhd"
    )

    exp_lkhd = (
        binom(n=4, p=0.75).logpmf(3)
        + binom(n=4, p=1 / 3).logpmf(1)
        + binom(n=2, p=1 / 3).logpmf(1)
        + binom(n=2, p=0.5).logpmf(1)
        + binom(n=3, p=0.4).logpmf(1)
        + binom(n=2, p=0.4).logpmf(1)
    )
    assert np.abs(lkhd - exp_lkhd) < 1e-6


def test_log_lkhd_evaluation(base_evaluator):
    lkhd = base_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="log_lkhd"
    )

    exp_lkhd = (
        binom(n=4, p=0.5).logpmf(1)
        + binom(n=4, p=0.5).logpmf(3)
        + binom(n=3, p=0.4).logpmf(1)
        + binom(n=2, p=0.4).logpmf(1)
    )
    assert np.abs(lkhd - exp_lkhd) < 1e-6


def test_loo_log_lkhd_evaluation(base_evaluator):
    lkhd = base_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="loo_log_lkhd"
    )

    exp_lkhd = (
        binom(n=4, p=0.75).logpmf(1)
        + binom(n=4, p=0.25).logpmf(3)
        + binom(n=3, p=0.5).logpmf(1)
        + binom(n=2, p=1 / 3).logpmf(1)
    )
    assert np.abs(lkhd - exp_lkhd) < 1e-6


def test_aic_evaluation(base_evaluator):
    aic = base_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="aic"
    )

    exp_lkhd = (
        binom(n=4, p=0.5).logpmf(1)
        + binom(n=4, p=0.5).logpmf(3)
        + binom(n=3, p=0.4).logpmf(1)
        + binom(n=2, p=0.4).logpmf(1)
    )

    assert np.abs(aic - (4 - 2 * exp_lkhd)) < 1e-6


def test_bic_evaluation(base_evaluator):
    bic = base_evaluator.evaluate(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
        metric="bic"
    )

    exp_lkhd = (
        binom(n=4, p=0.5).logpmf(1)
        + binom(n=4, p=0.5).logpmf(3)
        + binom(n=3, p=0.4).logpmf(1)
        + binom(n=2, p=0.4).logpmf(1)
    )

    assert np.abs(bic - (np.log(13) * 2 - 2 * exp_lkhd)) < 1e-6


def test_missing_sku_raise(base_evaluator):
    with pytest.raises(ValueError) as e:
        base_evaluator.evaluate(
            skus=["a", "c", "d"],
            cluster_idx=[0, 1, 1],
            metric="log_lkhd"
        )

    assert e.match("Sku b")


def test_mismatched_dimensions_raise(base_evaluator):
    with pytest.raises(ValueError) as e:
        base_evaluator.evaluate(
            skus=["a", "b", "c", "d"],
            cluster_idx=[0, 1, 1],
            metric="log_lkhd"
        )


def test_sku_meta_data(meta_data_evaluator):
    names = meta_data_evaluator.names(
        skus=["a", "b", "c", "d"],
        cluster_idx=[0, 0, 1, 1],
    )

    assert names == {
        0: ["name_a", "name_b"],
        1: ["name_c", "name_d"]
    }
