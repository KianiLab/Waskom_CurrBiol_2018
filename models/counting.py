import numpy as np
from scipy import stats, integrate

from .base import Model, merge_tables


class Counting(Model):

    param_names = ["sigma"]
    param_text = {"sigma": "σ_η"}
    color = "#635790"

    def simulate_dataset(self, n, data=None, seed=None):

        rs = np.random.RandomState(seed)

        # Generate the basic pulse-wise data
        trial_data, pulse_data = self.simulate_experiment(n, data, rs)
        n_trials = len(trial_data)
        n_pulses = len(pulse_data)

        # Add Gaussian noise to each pulse
        sigma = self.params["sigma"]
        noise = rs.normal(0, sigma, n_pulses)
        llr_obs = pulse_data["pulse_llr"] + noise

        # Determine the trialwise response
        dv = (pulse_data
              .assign(llr_sign=np.sign(llr_obs))
              .groupby(self.trial_grouper, sort=False)
              .llr_sign
              .sum())
        response = (dv > 0).astype(np.int)
        response = np.where(dv == 0, rs.choice([0, 1], n_trials), response)

        # Update the trial data structure
        trial_data["response"] = response.astype(int)
        trial_data["correct"] = response == trial_data["target"]

        # Merge the trial and pulse data structure
        pulse_data = merge_tables(pulse_data, trial_data)

        return trial_data, pulse_data

    def predict_response(self, trial_data, pulse_data):

        sigma = self.params.sigma

        def trial_func(trial):
            n = len(trial)
            ks = np.arange(n // 2 + 1, n + 1)
            ps = []
            for k in ks:
                ps.append(_p_n_k(trial.p_inc.values, k))
            ps.append(.5 * _p_n_k(trial.p_inc.values, n / 2))
            return np.sum(ps)

        cols = self.trial_grouper + ["pulse_llr"]
        fit_data = pulse_data[cols].copy()

        p_inc = stats.norm.sf(0, fit_data.pulse_llr, sigma)
        fit_data["p_inc"] = p_inc

        trial_p = (fit_data
                   .groupby(self.trial_grouper, sort=False)
                   .apply(trial_func))
        return trial_p

    def predict_evidence_func(self, xbar, lim=3, dx=.25):

        sigma = self.params.sigma

        def g(mesh):

            n = mesh.shape[1]
            ks = np.arange(n // 2 + 1, n + 1)
            G = np.zeros((mesh.shape[0], n + 1))

            pmesh = stats.norm.sf(0, mesh, sigma)

            for k in ks:
                G[:, k] = _p_n_k(pmesh.T, k)
            if n % 2 == 0:
                G[:, n // 2] = .5 * _p_n_k(pmesh.T, n / 2)

            return G.sum(axis=1)

        return self._predict_evidence_func_generic(g, xbar, lim, dx)

    def predict_sample_func(self, n=None):

        sigma = self.params.sigma

        design = self.design
        if n is None:
            n = design["count"]
        n = np.asarray(n)

        # Define probability of positive increment on any sample
        m, s = design["llr_m"], design["llr_sd"]
        d = stats.norm(m, np.sqrt(s ** 2 + sigma ** 2))
        p_inc = d.sf(0)

        # Use binomial distribution to predict accuracy
        B = stats.binom(n, p_inc)
        P = B.sf(n / 2) + .5 * B.pmf(n / 2)

        return P

    def predict_reverse_func_single(self, n):

        d = self.design["dh"]
        sigma = self.params.sigma

        # Get the generating distribution variance
        m, v = d.stats()

        # Define the distribution of sensory noise
        d_noise = stats.norm(0, sigma)

        # Compute the variance of the "observed" evidence (signal + noise)
        obs_v = v + sigma ** 2

        # Define probability of positive increment on any sample
        p_inc = stats.norm(m, np.sqrt(obs_v)).sf(0)

        # Find the marginal probabilities of correct and incorrect choices
        Bmarg = stats.binom(n, p_inc)
        P_C = Bmarg.sf(n / 2) + .5 * Bmarg.pmf(n / 2)
        P_W = 1 - P_C

        # Define functions to find the conditional probability of the
        # response given the generated evidence on each pulse

        def qint(f, a=-np.inf, b=np.inf, *args, **kwargs):
            """Wrapper function for integration to simplify code below."""
            return integrate.quad(f, a, b, *args, **kwargs)[0]

        B = stats.binom(n - 1, p_inc)
        P_C_g_Ci = B.sf(n / 2 - 1) + .5 * B.pmf(n / 2 - 1)
        P_W_g_Ci = 1 - P_C_g_Ci
        P_C_g_Wi = B.sf(n / 2) + .5 * B.pmf(n / 2)
        P_W_g_Wi = 1 - P_C_g_Wi

        if n > 1:
            if sigma > 0:
                def P_C_g_X(x):
                    d_obs = stats.norm(x, sigma)
                    return d_obs.sf(0) * P_C_g_Ci + d_obs.cdf(0) * P_C_g_Wi

                def P_W_g_X(x):
                    d_obs = stats.norm(x, sigma)
                    return d_obs.sf(0) * P_W_g_Ci + d_obs.cdf(0) * P_W_g_Wi
            else:
                def P_C_g_X(x):
                    return P_C_g_Ci if x > 0 else P_C_g_Wi

                def P_W_g_X(x):
                    return P_W_g_Ci if x > 0 else P_W_g_Wi

        else:
            if sigma > 0:
                def P_C_g_X(x):
                    return d_noise.sf(-x)

                def P_W_g_X(x):
                    return d_noise.cdf(-x)
            else:
                def P_C_g_X(x):
                    return float(x > 0)

                def P_W_g_X(x):
                    return float(x < 0)

        # Define the bounds for the outer integration, which have to be
        # special-cased for single-pulse trials when assuming no noise
        if n == 1 and sigma == 0:
            C_bounds, W_bounds = (0, np.inf), (-np.inf, 0)
        else:
            C_bounds = W_bounds = -np.inf, np.inf

        # Find the conditional expectation
        E_X_g_C = qint(lambda x: x * d.pdf(x) * P_C_g_X(x), *C_bounds) / P_C
        E_X_g_W = qint(lambda x: x * d.pdf(x) * P_W_g_X(x), *W_bounds) / P_W

        # The kernel will be constant over pulses, so expand the scalars
        # into vectors with the correct size and return
        return np.full(n, E_X_g_W).tolist(), np.full(n, E_X_g_C).tolist()


def _p_n_k(ps, k):
    n = len(ps)
    if k // 1 != k or k < 0 or n < k:
        return 0
    if n == 1:
        assert k in (0, 1)
        return ps[0] if k == 1 else 1 - ps[0]
    p = (_p_n_k(ps[:-1], k) * (1 - ps[-1])
         + _p_n_k(ps[:-1], k - 1) * ps[-1])
    return p
