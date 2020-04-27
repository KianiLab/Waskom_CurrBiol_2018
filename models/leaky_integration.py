import numpy as np
import pandas as pd
from scipy import stats

from .base import Model, merge_tables


class LeakyIntegration(Model):

    param_names = ["sigma_eta", "sigma_eps", "alpha"]
    param_text = {"sigma_eta": "σ_η", "sigma_eps": "σ_ϵ", "alpha": "α"}
    color = "#11875d"

    def simulate_dataset(self, n, data=None, seed=None, dt=.1):

        rs = np.random.RandomState(seed)

        # Generate the basic pulse-wise data
        trial_data, pulse_data = self.simulate_experiment(n, data, rs)
        n_trials = len(trial_data)
        n_pulses = len(pulse_data)

        # Add gaussian-distributed noise to each pulse
        sigma_eta = self.params.sigma_eta
        noise = rs.normal(0, sigma_eta, n_pulses)
        llr_obs = pulse_data.pulse_llr + noise

        # Reformat data object for fast simulation
        onset_idx = (pulse_data.pulse_time / dt).round().astype(np.int)
        choice_idx = (trial_data.trial_dur / dt).round().astype(np.int)

        index = trial_data.set_index(self.trial_grouper).index
        trial_idx = pulse_data.join(
            pd.Series(np.arange(n_trials), index, np.int, "trial_idx"),
            on=self.trial_grouper).trial_idx

        pulses = np.zeros((choice_idx.max(), n_trials))
        pulses[onset_idx.values, trial_idx.values] = llr_obs

        # Vectorize the leaky integration over trials
        sigma_eps = self.params.sigma_eps
        alpha = self.params.alpha
        V = np.zeros_like(pulses)
        diffusion_noise = rs.normal(0, sigma_eps, V.shape) * np.sqrt(dt)
        for t in range(choice_idx.max() - 1):
            leak = alpha * V[t] * dt
            dV = pulses[t] - leak + diffusion_noise[t]
            V[t + 1] = V[t] + dV

        # Determine choice by the sign of the accumulator
        response = V[choice_idx - 1, np.arange(n_trials)] > 0

        trial_data["response"] = response.astype(int)
        trial_data["correct"] = response == trial_data["target"]

        # Merge the trial and pulse data structure
        pulse_data = merge_tables(pulse_data, trial_data)

        return trial_data, pulse_data

    def predict_response(self, trial_data, pulse_data):

        var_c = self.params.sigma_eta ** 2
        var_d = self.params.sigma_eps ** 2
        alpha = self.params.alpha

        trial_data = trial_data.set_index(self.trial_grouper)
        pulse_data = pulse_data.set_index(self.trial_grouper + ["pulse"])

        llr = pulse_data.pulse_llr.unstack().fillna(0)
        gap = pulse_data.gap_dur.unstack().fillna(0)
        var = var_c * pulse_data.assign(v=1.0).v.unstack().fillna(0)

        t0 = trial_data.wait_pre_stim
        m, v = ornstein_uhlenbeck_moments(t0, 0, 0, alpha, var_d)

        for i in llr:
            m += llr[i].values
            v += var[i].values
            t = gap[i].values
            m, v = ornstein_uhlenbeck_moments(t, m, v, alpha, var_d)

        trial_p = pd.Series(stats.norm(m, np.sqrt(v)).sf(0), llr.index)
        return trial_p.reindex(trial_data.index)

    def predict_evidence_func(self, xbar):

        raise NotImplementedError

    def predict_sample_func(self, n=None):

        raise NotImplementedError

    def predict_reverse_func_single(self, n):

        raise NotImplementedError

    def summary(self, full=None):

        return self._brief_summary()


def ornstein_uhlenbeck_moments(t, m0, v0, alpha, var):

    if alpha == 0:
        return m0, v0 + var * t

    m = m0 * np.exp(-alpha * t)
    v = (var / (2 * alpha) * (1 - np.exp(-t * 2 * alpha))
         + v0 * np.exp(-t * 2 * alpha))
    return m, v
