import numpy as np
from scipy import stats, integrate

from .base import Model, merge_tables


class LinearIntegration(Model):

    param_names = ["sigma"]
    param_text = {"sigma": "σ_η"}
    color = "#265EA6"

    def simulate_dataset(self, n, data=None, seed=None):

        rs = np.random.RandomState(seed)

        # Generate the basic pulse-wise data
        trial_data, pulse_data = self.simulate_experiment(n, data, rs)
        n_pulses = len(pulse_data)

        # Add Gaussian noise to each pulse
        sigma = self.params.sigma
        noise = rs.normal(0, sigma, n_pulses)
        llr_obs = pulse_data["pulse_llr"] + noise

        # Compute the trial-wise decision variable and simulate the response
        dv = (pulse_data
              .assign(llr_obs=llr_obs)
              .groupby(self.trial_grouper, sort=False)
              .llr_obs
              .sum())
        response = np.where(dv > 0, 1, 0)
        trial_data["response"] = response.astype(int)
        trial_data["correct"] = response == trial_data["target"]

        # Merge the trial and pulse data structure
        pulse_data = merge_tables(pulse_data, trial_data)

        return trial_data, pulse_data

    def predict_response(self, trial_data, pulse_data):

        sigma = self.params.sigma
        dv_mean = (pulse_data
                   .groupby(self.trial_grouper, sort=False)
                   .pulse_llr
                   .sum())
        dv_std = np.sqrt(sigma ** 2 * trial_data["pulse_count"])
        return stats.norm(dv_mean, dv_std).sf(0)

    def predict_evidence_func(self, xbar):

        xbar = np.asarray(xbar)
        sigma = self.params.sigma
        design = self.design
        pmfs = [
            stats.norm.sf(0, xbar, sigma / np.sqrt(n)) for n in design["count"]
        ]
        pmf = np.average(pmfs, axis=0, weights=design["count_pmf"])
        return pmf

    def predict_sample_func(self, n=None):

        sigma = self.params.sigma
        design = self.design
        if n is None:
            n = design["count"]
        n = np.asarray(n)

        m_x, s_x = design["llr_m"], design["llr_sd"]
        d = stats.norm(m_x * n, np.sqrt((s_x ** 2 + sigma ** 2) * n))
        f = d.sf(0)

        return f

    def predict_reverse_func_single(self, n):

        d = self.design["dh"]
        sigma = self.params.sigma

        # Get the generating distribution variance
        m, v = d.stats()

        # Compute the variance of the "observed" evidence (signal + noise)
        obs_v = v + sigma ** 2

        # Define normal distribution object for the noise
        d_noise = stats.norm(0, sigma)

        # Define the distribution of total evidence on the other pulses
        d_other = stats.norm(m * (n - 1), np.sqrt(obs_v * (n - 1)))

        # Find the marginal probabilities of correct and incorrect choices
        d_resp = stats.norm(m * n, np.sqrt(obs_v * n))
        P_C = d_resp.sf(0)
        P_W = d_resp.cdf(0)

        # Define functions to find the conditional probability of the
        # response given the generated evidence on each pulse

        def qint(f, a=-np.inf, b=np.inf, *args, **kwargs):
            """Wrapper function for integration to simplify code below."""
            return integrate.quad(f, a, b, *args, **kwargs)[0]

        if n > 1:
            if sigma > 0:
                def P_C_g_X(x):
                    return qint(lambda v: d_noise.pdf(v) * d_other.sf(-v - x),
                                -10 * sigma, 10 * sigma)

                def P_W_g_X(x):
                    return qint(lambda v: d_noise.pdf(v) * d_other.cdf(-v - x),
                                -10 * sigma, 10 * sigma)
            else:
                def P_C_g_X(x):
                    return d_other.sf(-x)

                def P_W_g_X(x):
                    return d_other.cdf(-x)
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
            C_bounds, W_bounds = (0, 10 * obs_v), (-10 * obs_v, 0)
        else:
            C_bounds = W_bounds = -10 * obs_v, 10 * obs_v

        # Find the conditional expectation
        E_X_g_C = qint(lambda x: x * d.pdf(x) * P_C_g_X(x), *C_bounds) / P_C
        E_X_g_W = qint(lambda x: x * d.pdf(x) * P_W_g_X(x), *W_bounds) / P_W

        # The kernel will be constant over pulses, so expand the scalars
        # into vectors with the correct size and return
        return np.full(n, E_X_g_W).tolist(), np.full(n, E_X_g_C).tolist()
