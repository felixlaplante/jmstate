"""Tests for MCMC sampling."""

from jmstate.types._data import ModelDataUnchecked

from ._helpers import _data, _model


def test_step():
    model, data = _model(tol=0.0), _data()
    prepared = ModelDataUnchecked(
        data.x, data.t, data.y, data.trajectories, data.c
    ).prepare(model)
    sampler = model._init_sampler(prepared)
    sampler.step().run(1)
    assert len(sampler.diagnostics_["mean_accept_rate"]) == 2
