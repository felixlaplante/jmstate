"""Tests for model fitting."""

import math
import warnings

import pytest
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import InvalidParameterError

from ._helpers import _data, _model


def test_warning():
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        _model(max_iter=1, window_size=3).fit(_data())


def test_optimizer():
    with pytest.raises(ValueError, match="Optimizer is not initialized"):
        _model(fit=False).fit(_data())


def test_parameters():
    with pytest.raises(InvalidParameterError, match="max_iter"):
        _model(max_iter=-1).fit(_data())


def test_convergence():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model().fit(_data())
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]


def test_summary():
    torch.manual_seed(42)
    model = _model()
    model.n_subsample = 1
    model.fit(_data())

    with pytest.raises(TypeError):
        model.compute_summary(2, 4, 2)

    model.compute_summary(
        n_posterior_samples=8,
        n_importance_samples=4,
        importance_batch_size=2,
    )
    assert math.isfinite(model.loglik_)
