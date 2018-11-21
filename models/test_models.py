"""Simple smoke tests for modeling code."""
import os
import numpy as np
import pandas as pd

import pytest
from . import LinearIntegration, ExtremaDetection, Counting, LeakyIntegration


RANDOM_SEED = 112118


def test_experiment_simulation():

    m = LinearIntegration()

    n = 50
    trial_sim, pulse_sim = m.simulate_experiment(n)

    assert len(trial_sim) == n
    assert len(pulse_sim["trial"].unique()) == n

    n_rep = 2

    trial_rep, pulse_rep = m.simulate_experiment(n_rep, (trial_sim, pulse_sim))
    assert len(trial_rep) == (n * n_rep)
    assert len(pulse_rep["trial"].unique()) == n
    assert len(pulse_rep["subject"].unique()) == n_rep


@pytest.mark.parametrize(
    "model, params",
    [
        (LinearIntegration, {"sigma": .4}),
        (ExtremaDetection, {"sigma": .3, "theta": .7}),
        (Counting, {"sigma": .4}),
        (LeakyIntegration, {"sigma_eta": .4, "sigma_eps": .05, "alpha": .1}),
    ],
)
def test_model_basics(model, params):

    m = model(**params)
    assert m.params.to_dict() == params

    n_sim = 50
    trial_sim, pulse_sim = m.simulate_dataset(n_sim, seed=RANDOM_SEED)
    assert len(trial_sim) == n_sim

    p_r = m.predict_response(trial_sim, pulse_sim)
    assert len(p_r) == n_sim

    m.fit_parameters(trial_sim, pulse_sim, list(params.values()))


@pytest.mark.parametrize(
    "model, params",
    [
        (LinearIntegration, {"sigma": .4}),
        (ExtremaDetection, {"sigma": .3, "theta": .7}),
        (Counting, {"sigma": .4}),
    ],
)
def test_model_assays(model, params):

    m = model(**params)

    xbar = np.arange(-1.25, 1.5, .25)
    f = m.predict_evidence_func(xbar)
    assert len(f) == len(xbar)

    pulses = np.arange(1, 6)
    f = m.predict_sample_func(pulses)
    assert len(f) == len(pulses)

    for n in [1, 2]:
        f_w, f_c = m.predict_reverse_func_single(n)
        assert len(f_w) == n
        assert len(f_c) == n
        assert np.greater(f_c, f_w).all()


@pytest.mark.parametrize(
    "model,init_params",
    [
        (LinearIntegration, [.3]),
        (ExtremaDetection, [.3, .8]),
        (Counting, [.3]),
        (LeakyIntegration, [.3, .05, .05]),
    ],
)
def test_data_fits(model, init_params):

    code_dir = os.path.dirname(__file__)
    trial_data = pd.read_csv(f"{code_dir}/../data/trial_data.csv")
    pulse_data = pd.read_csv(f"{code_dir}/../data/pulse_data.csv")

    m = model()
    m.fit_parameters(trial_data, pulse_data, init_params)
