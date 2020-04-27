import numpy as np
import pandas as pd
from scipy import stats, optimize


TRIAL_GROUPER = ["subject", "timing", "session", "run", "trial"]


class Model(object):
    """Base class for other models to derive from."""
    def __init__(self, gap_params=None, fix=None, trial_grouper=TRIAL_GROUPER,
                 **params):

        self.design = design_parameters(gap_params)
        self.trial_grouper = trial_grouper

        unexpected_params = set(params) - set(self.param_names)
        if unexpected_params:
            err = f"The following parameters do not exist: {unexpected_params}"
            raise ValueError(err)

        all_params = {k: params.get(k, None) for k in self.param_names}
        self.params = ParamSet(initial=all_params, fix=fix,
                               order=self.param_names)

        self._predictions = dict()

    def simulate_experiment(self, n, data=None, random_state=None):
        """Simulate basic information for each pulse."""
        if random_state is None:
            rs = np.random.RandomState()
        else:
            rs = random_state

        trial_cols = [
            "target", "pulse_count", "trial_dur", "wait_pre_stim",
        ]
        pulse_cols = [
            "pulse", "pulse_llr", "pulse_time", "pulse_dur", "gap_dur",
        ]

        if data is not None:

            trial_keep = self.trial_grouper + trial_cols
            pulse_keep = self.trial_grouper + pulse_cols

            trial_data, pulse_data = data
            trial_all = []
            pulse_all = []

            for i in range(n):
                trial_i = trial_data[trial_keep].copy()
                pulse_i = pulse_data[pulse_keep].copy()
                trial_i["subject"] += f"_sim{i:02d}"
                pulse_i["subject"] += f"_sim{i:02d}"
                trial_all.append(trial_i)
                pulse_all.append(pulse_i)

            trial_data = pd.concat(trial_all, ignore_index=True)
            pulse_data = pd.concat(pulse_all, ignore_index=True)
            return trial_data, pulse_data

        design = self.design

        # Sample the pulse count for each trial
        count = rs.choice(design["count"], n, p=design["count_pmf"])

        # Define trial and pulse lables
        trial = np.arange(1, n + 1)
        pulse = np.concatenate([np.arange(1, c + 1) for c in count])
        pulse_trial = np.concatenate([
            np.full(c, i) for i, c in enumerate(count, 1)
        ])

        # Define the "target" for each trial (the generating distribution)
        trial_target = rs.choice([0, 1], n)
        pulse_target = np.concatenate([
            np.full(c, t) for t, c in zip(trial_target, count)
        ])

        # Sample the LLR for each pulse
        pulse_llr = rs.normal(design["llr_m"], design["llr_sd"], count.sum())
        pulse_llr[pulse_target == 0] *= -1

        # Sample the pulse gaps
        gap_dur = self.design["gap_dist"].rvs(count.sum(), random_state=rs)
        wait_pre_stim = design["gap_dist"].rvs(n, random_state=rs)

        # Construct the trial-wise data table
        trial_data = pd.DataFrame(
            dict(trial=trial, target=trial_target, pulse_count=count,
                 wait_pre_stim=wait_pre_stim),
            columns=self.trial_grouper + trial_cols,
        )

        # Construct the initial pulse-wise data table
        pulse_data = pd.DataFrame(
            dict(trial=pulse_trial, pulse=pulse, pulse_llr=pulse_llr,
                 gap_dur=gap_dur, pulse_dur=.2, occurred=True),
            columns=self.trial_grouper + pulse_cols
        )

        # Add in the pulse time information
        def pulse_time_func(x):
            return x.shift(1).fillna(0).cumsum()
        trial_times = (pulse_data.gap_dur
                       + pulse_data.pulse_dur
                       ).groupby(pulse_data.trial)
        pulse_time = (np.repeat(wait_pre_stim, count)
                      + trial_times.transform(pulse_time_func).values)
        pulse_data["pulse_time"] = pulse_time

        # Add in the trial time information
        trial_dur = trial_data.wait_pre_stim + trial_times.sum().values
        trial_data["trial_dur"] = trial_dur

        # Add in dummy identifiers to match real data structure
        for data in [trial_data, pulse_data]:
            for col in [c for c in self.trial_grouper if c != "trial"]:
                data[col] = "sim"

        return trial_data, pulse_data

    def simulate_dataset(self, n, seed):
        """Simulate the decision process over many trials.

        Must be defined by a sub-class.

        """
        raise NotImplementedError

    def fit_parameters(self, trial_data, pulse_data, p0,
                       verbose=False, tol=None):
        """Main interface to maximum likelihood estimation of parameters."""
        self.fit_data = trial_data, pulse_data

        p0 = self._pack_fit_params(p0)

        def errfunc(p):

            p = self.params.update(self._unpack_fit_params(p))
            trial_p = self.predict_response(trial_data, pulse_data)
            ll = self.bernoulli_loglike(trial_data.response, trial_p)
            self._print_opt_values(verbose, p, ll)
            return -ll

        if tol is None:
            try:
                tol = self.converge_tol
            except AttributeError:
                tol = .0001

        res = optimize.minimize(errfunc, p0, method="Nelder-Mead", tol=tol)
        self.params.update(self._unpack_fit_params(res.x))
        self.fit_result = res
        return res

    def _pack_fit_params(self, x):
        """Transform paramaters from evaluation space to optimization space."""
        return np.log(x)

    def _unpack_fit_params(self, x):
        """Transform parameters from optimization space to evaluation space."""
        return np.exp(x)

    def _predict_evidence_func_generic(self, g, xbar, lim=3, dx=.25):
        """General function for doing a grid-approximation for the PMF."""
        key = ("evidence", xbar.data.hex(), lim, dx, self.params.hex)
        if key in self._predictions:
            return self._predictions[key]

        dh, dl = self.design["dh"], self.design["dl"]

        # Define the (one-dimensional) mesh for grid sampling
        xx = np.arange(-lim, lim + dx, dx)

        # Initialize the output, which we will build up additively
        pmf = np.zeros_like(xbar)
        for n, p_n in zip(self.design["count"], self.design["count_pmf"]):

            # Make the n-dimensional mesh for this pulse count
            mesh = np.column_stack(a.flat for a in np.meshgrid(*[xx] * n))

            # Compute the sum of the evidence across pulses
            X = mesh.sum(axis=1)

            # Compute probability of choosing "high" for each stim sequence
            G = g(mesh)

            # Define indicator function to select sequences by mean evidence
            def I(w):  # noqa
                return np.abs(X / n - w) < (dx / 2)

            for d in [dh, dl]:

                # Compute the probability of each pulse sequence
                # P = d.pdf(mesh).prod(axis=1) * dx ** n  # too mem intensive?
                P = np.product([d.pdf(X_i)
                                for X_i in mesh.T], axis=0) * dx ** n

                # Compute the psychometric function across bins and weight by
                # probability of the sample count and generating distribution
                for i, w in enumerate(xbar):
                    Z = (P * I(w)).sum()
                    pmf[i] += .5 * p_n * (P * G * I(w)).sum() / Z

        self._predictions[key] = pmf
        return pmf

    def predict_evidence_func(self, xbar):
        """Function relating responses to mean pulse strength.

        Must be defined by a sub-class.

        """
        raise NotImplementedError

    def predict_sample_func(self, n):
        """Function relating accuracy to pulse count.

        Must be defined by a sub-class.

        """
        raise NotImplementedError

    def predict_reverse_func(self, align, counts=None, count_pmf=None):
        """Function estimating conditional estimate of evidence.

        Sub-classes should define count-specific estimation functions.

        """
        design = self.design
        if counts is None:
            counts = design["count"]
            count_pmf = design["count_pmf"]
        if np.isscalar(counts):
            counts = [counts]
            count_pmf = [1]
        counts = np.asarray(counts)
        count_pmf = np.asarray(count_pmf)

        # Define weighting function to compute the kernel weight at each pulse
        pulses = np.arange(counts.max()) + 1
        pulse_weights = np.zeros((len(counts), counts.max()))
        for i, n in enumerate(pulses):
            idx = max(0, n - counts.min())
            pulse_weights[idx:, i] = count_pmf[idx:]
        pulse_weights /= pulse_weights.sum(axis=0)

        # Initialize the data structures for the full kernels
        E_X_g_W = np.zeros(counts.max())
        E_X_g_C = np.zeros(counts.max())

        # Loop over individual pulse counts and predict the kernel for
        # trials with that count, then weight and add to the full kernel
        for i, n in enumerate(counts):

            W = pulse_weights[i, :n]

            key = ("reverse", self.params.hex, n)
            try:
                val = self._predictions[key]
            except KeyError:
                val = self.predict_reverse_func_single(n)
                self._predictions[key] = val
            E_X_g_W_N, E_X_g_C_N = val

            if align == "start":
                slc = slice(0, n)
            elif align == "end":
                slc = slice(-n, counts.max())
                W = W[::-1]

            E_X_g_W[slc] += W * E_X_g_W_N
            E_X_g_C[slc] += W * E_X_g_C_N

        return E_X_g_W, E_X_g_C

    def summary(self, full=True):
        """Return information about fit and predictions."""
        if full:
            return self._full_summary()
        else:
            return self._brief_summary()

    def _brief_summary(self):
        """Summary only of fit results."""
        trial_data, pulse_data = self.fit_data

        return dict(
            params=self.params.to_dict(),
            loglike=-self.fit_result.fun,
            success=self.fit_result.success,
            n_trials=len(trial_data),
            n_pulses=len(pulse_data),
        )

    def _full_summary(self):
        """Summary of fit results and behavioral assay predictions."""
        summary = self._brief_summary()

        trial_data, pulse_data = self.fit_data

        # Evidence psychometric function
        pmf_limit, pmf_step = 1.25, .25
        xbar = np.arange(-pmf_limit, pmf_limit + pmf_step, pmf_step)
        model_epmf = self.predict_evidence_func(xbar)

        # Sample psychometric function
        model_spmf = self.predict_sample_func()

        # Reverse correlation
        model_kernels = []
        for n in self.design["count"]:
            model_kernels.extend(self.predict_reverse_func("start", n))
        model_kernel = np.concatenate(model_kernels)

        reverse_func = (
            [list(f) for f in self.predict_reverse_func("start")],
            [list(f) for f in self.predict_reverse_func("end")]
        )

        summary.update(
            xbar=list(xbar),
            evidence_func=list(model_epmf),
            sample_func=list(model_spmf),
            reverse_func=reverse_func,
            reverse_points=list(model_kernel),
        )
        return summary

    def bernoulli_loglike(self, r, p):
        """Log likelihood of responses given Bernoulli probabilities."""
        eps = np.finfo(np.float).eps
        p = np.clip(p, eps, 1 - eps)
        loglike = np.where(r, np.log(p), np.log(1 - p)).sum()
        return loglike

    def crossval_loglike(self, trial_data, pulse_data, *args, **kwargs):
        """Interface for fitting and getting likelihood across CV splits."""
        trial_labels = trial_data["timing"] + trial_data["session"].astype(str)
        pulse_labels = pulse_data["timing"] + pulse_data["session"].astype(str)

        label_set = trial_labels.unique()

        loglike = 0
        for label in label_set:

            trial_train = trial_data[trial_labels != label]
            pulse_train = pulse_data[pulse_labels != label]

            trial_test = trial_data[trial_labels == label]
            pulse_test = pulse_data[pulse_labels == label]

            self.fit_parameters(trial_train, pulse_train, *args, **kwargs)

            pred = self.predict_response(trial_test, pulse_test)
            loglike += self.bernoulli_loglike(trial_test["response"], pred)

        return loglike

    def _print_opt_values(self, verbose, params, logL):
        """Function for printing information about ongoing optimization."""
        if verbose and not self.params.iter % verbose:
            s = "{:>4} | ".format(self.params.iter)
            for name in self.param_names:
                val = params[name]
                text = self.param_text.get(name, name)
                s += "{}: {:.2f} | ".format(text, val)
            s += "logL: {:.2f}".format(logL)
            print(s)


class ParamSet(object):
    """Object for managing model parameters during model fitting.

    The main contribution of this class is separation of free and fixed
    parameters in a way that works with scipy optimize functionality.
    Parameters can be accessed through the `.params` attribute, in the separate
    `.free` and `.fixed` attributes, or directly by names. Free parameters can
    be updated with an array that lacks semantic information (what scipy uses
    internally) and are mapped properly to the named parameters.

    """
    def __init__(self, initial, fix=None, order=None):
        """Set initial values and determine fixed parameters.

        Parameters
        ----------
        initial : Series or dictionary
            Initial values for parameters.
        fix : list of strings, optional
            Names of parameters to fix at initial values.
        order : list of strings, optional
            Order of paramters in the series

        """
        if isinstance(initial, dict):
            initial = pd.Series(initial, order)

        self.names = list(initial.index)
        self.params = initial.copy()

        self.iter = 0

        if fix is None:
            fix = []

        if set(fix) - set(self.names):
            raise ValueError("Fixed parameters must appear in `initial`")

        self.fixed_names = [n for n in self.names if n in fix]
        self.free_names = [n for n in self.names if n not in fix]

    def __repr__(self):
        """Show the values and fixed status of each parameter."""
        s = ""
        s += "Free Parameters:\n"
        for name, val in self.free.iteritems():
            s += "  {}: {:.3g}\n".format(name, val)
        if self.fixed_names:
            s += "Fixed Parameters:\n"
            for name, val in self.fixed.iteritems():
                s += "  {}: {:.3g}\n".format(name, val)
        return s

    def __getattr__(self, name):
        """Allow dot access to params."""
        if name in self.params:
            return self.params[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, val):
        """Allow dot access to params."""
        object.__setattr__(self, name, val)

    def __getitem__(self, name):
        """Allow bracket access to params."""
        return self.params[name]

    def __setitem__(self, name, val):
        """Allow bracket access to params."""
        self.params[name] = val

    @property
    def free(self):
        """Return a vector of current free parameter values."""
        return self.params[self.free_names]

    @property
    def fixed(self):
        """Return a vector of current fixed parameter values."""
        return self.params[self.fixed_names]

    @property
    def hex(self):
        """Hex values of current parameter values."""
        return self.params.values.data.hex()

    def to_dict(self):
        """Return current parameters as a dictionary."""
        return self.params.to_dict()

    def update(self, params):
        """Set new values for the free parameters and return self.

        Parameters
        ----------
        params : ParamSet, Series, dictionary, or vector
            Either an equivalent ParamSet, Series or dictionary mapping
            parameter names to values, or a vector of parameters in the order
            of `self.free`.

        Returns
        -------
        self : ParamSet
            Returns self with new parameter values.

        """
        if isinstance(params, ParamSet):
            if params.free_names != self.free_names:
                err = "Input object must have same free parameters."
                raise ValueError(err)
            self.params.update(params.params)

        elif isinstance(params, pd.Series):
            if list(params.index) != self.free_names:
                err = "Input object must have same free parameters."
                raise ValueError(err)
            self.params.update(params)

        elif isinstance(params, dict):
            if set(params) - set(self.free_names):
                err = "Input object has unknown parameters"
                raise ValueError(err)
            elif set(self.free_names) - set(params):
                err = "Input object is missing parameters"
                raise ValueError(err)
            self.params.update(pd.Series(params))

        elif isinstance(params, (np.ndarray, list, tuple)):
            if len(params) != len(self.free_names):
                err = "Input object has wrong number of parameters."
                raise ValueError(err)
            new_params = pd.Series(params, self.free_names)
            self.params.update(new_params)

        else:
            err = "Type of `values` is not understood"
            raise ValueError(err)

        self.iter += 1

        return self


def design_parameters(gap_params=None):
    """Generate a dictionary with default design parameters."""
    # Distributions of pulses per trial
    count = [1, 2, 3, 4, 5]
    count_pmf = trunc_geom_pmf(count, .25)

    # Distribution parameters in stimulus units
    means = -1.1, -0.9
    sd = .15

    # Distributions in log-likelihood ratio units
    llr_m, llr_sd = params_to_llr(means, sd)
    dh, dl = stats.norm(+llr_m, llr_sd), stats.norm(-llr_m, llr_sd)

    # Pulse gap duration
    if gap_params is None:
        gap_params = 3, 2, 2
    gap_dist = stats.truncexpon(*gap_params)

    # Design dictionary to pass to functions
    design = dict(count=count, count_pmf=count_pmf,
                  means=means, sds=sd, llr_m=llr_m, llr_sd=llr_sd,
                  dh=dh, dl=dl, gap_dist=gap_dist)

    return design


def params_to_llr(means, sd):
    """Convert gaussian distribution parameters to LLR units."""
    d1 = stats.norm(means[1], sd)
    d0 = stats.norm(means[0], sd)

    x = means[1]
    llr_m = np.log10(d1.pdf(x)) - np.log10(d0.pdf(x))
    llr_sd = np.log10(d1.pdf(x + sd)) - np.log10(d0.pdf(x + sd)) - llr_m

    return llr_m, llr_sd


def trunc_geom_pmf(support, p):
    """Define the PMF for a truncated geometric distribution."""
    a, b = min(support) - 1, max(support)
    dist = stats.geom(p=p, loc=a)
    return list(dist.pmf(support) / (dist.cdf(b) - dist.cdf(a)))


def merge_tables(pulse_data, trial_data, merge_keys=TRIAL_GROUPER):
    """Add trial-wise information to the pulse-wise table."""
    pulse_data = pulse_data.merge(trial_data, on=merge_keys)
    add_kernel_data(pulse_data)
    return pulse_data


def add_kernel_data(pulse_data):
    """Add variables that are useful for reverse correlation analysis."""
    pulse_data["kernel_llr"] = np.where(pulse_data.target == 1,
                                        pulse_data.pulse_llr,
                                        -1 * pulse_data.pulse_llr)

    pulse_data["pulse_start"] = pulse_data["pulse"]
    pulse_data["pulse_end"] = (pulse_data["pulse"]
                               - pulse_data["pulse_count"]
                               - 1)
    return pulse_data
