import numpy as np
import pandas as pd
from scipy import stats, integrate

from .base import Model, merge_tables


class ExtremaDetection(Model):

    param_names = ["sigma", "theta"]
    param_text = {"sigma": "σ_η", "theta": "θ_x"}

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

        # Identify pulses that exceed the thresholds
        theta = self.params["theta"]
        extrema = np.take([np.nan, 0, np.nan, 1],
                          np.digitize(llr_obs,
                                      [-np.inf, -theta, +theta, np.inf]))

        # Compute the trial-wise response following extrema detection
        trial_index = trial_data.set_index(self.trial_grouper).index
        random_guess = pd.Series(rs.choice([0, 1], n_trials), trial_index)
        response = (
            pulse_data
            .assign(extrema=extrema)
            .dropna()
            .groupby(self.trial_grouper, sort=False)
            .extrema
            .apply(lambda s: s.iloc[0])
            .reindex(trial_data[self.trial_grouper])
            .fillna(random_guess)
            .rename("response")
            .astype(np.int)
            .reset_index(drop=True)
        )

        trial_data["response"] = response.astype(int)
        trial_data["correct"] = response == trial_data["target"]

        # Merge the trial and pulse data structure
        pulse_data = merge_tables(pulse_data, trial_data)

        return trial_data, pulse_data

    def predict_response(self, trial_data, pulse_data):

        sigma = self.params.sigma
        theta = self.params.theta

        llr = pulse_data.pulse_llr
        cols = self.trial_grouper + ["pulse"]
        ix = pulse_data.set_index(cols).index

        norm = stats.norm(llr, sigma)

        # Prob of exceeding the 'high' threshold
        p_h = pd.Series(norm.sf(+theta), ix)

        # Prob of not exceeding either threshold
        p_m = pd.Series(norm.sf(-theta), ix) - p_h

        # Prob of not having committed at end of trial
        p_u = p_m.groupby(level=self.trial_grouper).cumprod()

        # Prob of not having committed at start of trial
        p_w = p_u.groupby(level=self.trial_grouper).shift(1).fillna(1)

        # Probability of never exceeding a threshold
        p_g = p_u.groupby(level=self.trial_grouper).min()

        # Probability of responding 'high'
        p = (p_h * p_w).groupby(level=self.trial_grouper).sum() + p_g * .5

        return p

    def predict_evidence_func(self, xbar, lim=3, dx=.25):

        sigma = self.params.sigma
        theta = self.params.theta

        def g(mesh):
            if sigma > 0:
                p_r = stats.norm.sf(+theta, mesh, sigma)
                p_w = stats.norm.sf(-theta, mesh, sigma) - p_r
                p_w = np.cumprod(p_w, axis=1)
                G = (p_r[:, 0]
                     + (p_r[:, 1:] * p_w[:, :-1]).sum(axis=1)
                     + p_w[:, -1] * .5)
            else:
                G = np.full(len(mesh), .5)
                for x_i in mesh.T:
                    G[(G == .5) & (x_i > +theta)] = 1
                    G[(G == .5) & (x_i < -theta)] = 0
            return G

        return self._predict_evidence_func_generic(g, xbar, lim, dx)

    def predict_sample_func(self, n=None):

        sigma = self.params.sigma
        theta = self.params.theta

        design = self.design
        if n is None:
            n = design["count"]
        n = np.asarray(n)

        # Define the observed evidence distribution
        m, s = design["llr_m"], design["llr_sd"]
        d = stats.norm(m, np.sqrt(s ** 2 + sigma ** 2))

        # Define the probability of seeing high or low extremum on each trial
        p_l, p_h = d.cdf(-theta), d.sf(+theta)

        # Define the probability of not seeing an extremum on each trial
        p_w = 1 - (p_l + p_h)

        # For each trial count, define the probability of positive response
        p_resps = []
        for c in n:
            pulses = np.arange(c)
            p = (p_w ** pulses * p_h).sum() + p_w ** c * .5
            p_resps.append(p)
        f = np.array(p_resps)

        return f

    def predict_reverse_func_single(self, n):

        def qint(f, a=-np.inf, b=np.inf, *args, **kwargs):
            """Wrapper function for integration to simplify code below."""
            return integrate.quad(f, a, b, *args, **kwargs)[0]

        d = self.design["dh"]
        sigma = self.params.sigma
        theta = self.params.theta

        # Combine signal and noise distributions to get "observed" distribution
        m, v = d.stats()
        d_obs = stats.norm(m, np.sqrt(v + sigma ** 2))
        d_noise = stats.norm(0, sigma)

        # Compute the probability of observing a value above, between, or below
        # the thresholds

        P_Eh = d_obs.sf(+theta)
        P_Em = d_obs.cdf(+theta) - d_obs.cdf(-theta)
        P_El = d_obs.cdf(-theta)

        assert np.allclose(P_Eh + P_Em + P_El, 1)

        # Define the probability of correct/wrong responses conditional on not
        # having committed at each pulse number

        pulses = np.arange(n)

        P_C = np.array([(P_Eh * P_Em ** np.arange(n)).sum() + .5 * P_Em ** n
                       for n in reversed(pulses + 1)])

        P_W = np.array([(P_El * P_Em ** np.arange(n)).sum() + .5 * P_Em ** n
                        for n in reversed(pulses + 1)])

        assert np.allclose(P_C + P_W, np.ones(n))

        # Define probability of correct or wrong response depending on seeing a
        # value that exceeds either of the thresholds if not committed

        P_C_g_Eh = np.ones(n)
        P_C_g_El = np.zeros(n)

        P_W_g_El = np.ones(n)
        P_W_g_Eh = np.zeros(n)

        # Define the probability of correct or wrong response conditional on
        # observing an intermediate value when not committed on each pulse

        P_C_g_Em = np.array([
            np.sum(P_Em ** i * P_Eh for i in range(n)) + P_Em ** n * .5
            for n in reversed(pulses)
        ])

        P_W_g_Em = np.array([
            np.sum(P_Em ** i * P_El for i in range(n)) + P_Em ** n * .5
            for n in reversed(pulses)
        ])

        # Get probability of seeing a value from each segment of distribution
        # conditional on eventually responding correctly or incorrectly

        P_Eh_g_C = P_Eh * P_C_g_Eh / P_C
        P_Em_g_C = P_Em * P_C_g_Em / P_C
        P_El_g_C = P_El * P_C_g_El / P_C

        assert np.array_equal(P_El_g_C, np.zeros(n))

        P_Eh_g_W = P_Eh * P_W_g_Eh / P_W
        P_Em_g_W = P_Em * P_W_g_Em / P_W
        P_El_g_W = P_El * P_W_g_El / P_W

        assert np.array_equal(P_Eh_g_W, np.zeros(n))

        # Compute the probability of not being committed at each pulse
        # given that you eventually respond either correctly or incorrectly

        P_unc_g_C = np.append(1, P_Em_g_C[:-1].cumprod())
        P_unc_g_W = np.append(1, P_Em_g_W[:-1].cumprod())

        # Compute the expected value of the generated evidence given your
        # probability of observing different kinds of events at each pulse

        def f_X_g_Eh(x):
            return x * d.pdf(x) * d_noise.sf(+theta - x)

        def f_X_g_Em(x):
            return x * d.pdf(x) * (d_noise.cdf(+theta - x)
                                   - d_noise.cdf(-theta - x))

        def f_X_g_El(x):
            return x * d.pdf(x) * d_noise.cdf(-theta - x)

        if sigma > 0:
            E_X_g_Eh = qint(f_X_g_Eh) / P_Eh
            E_X_g_Em = qint(f_X_g_Em) / P_Em
            E_X_g_El = qint(f_X_g_El) / P_El
        else:
            E_X_g_Eh = (qint(lambda x: x * d.pdf(x), theta, np.inf)
                        / d.sf(theta))
            E_X_g_Em = (qint(lambda x: x * d.pdf(x), -theta, theta)
                        / (d.cdf(theta) - d.cdf(-theta)))
            E_X_g_El = (qint(lambda x: x * d.pdf(x), -np.inf, -theta)
                        / d.cdf(-theta))

        E_X_g_C = (P_unc_g_C * (P_Eh_g_C * E_X_g_Eh + P_Em_g_C * E_X_g_Em)
                   + (1 - P_unc_g_C) * m)

        E_X_g_W = (P_unc_g_W * (P_El_g_W * E_X_g_El + P_Em_g_W * E_X_g_Em)
                   + (1 - P_unc_g_W) * m)

        return E_X_g_W.tolist(), E_X_g_C.tolist()
