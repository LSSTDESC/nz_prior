import numpy as np
from getdist import plots, MCSamples
from scipy.stats import multivariate_normal as mvn
from scipy.stats import kstest
import qp


class PriorBase:
    """
    Base class for priors. Projectors are used to project the measured
    photometric distributions by RAIL onto the space of a given generative
    photometric model for inference.
    This class is not meant to be used directly,
    but to be subclassed by specific projectors.
    The subclasses should implement the following methods:
    - evaluate_model: given a set of parameters, evaluate the model
    - get_prior: return the prior distribution of the model given
    the meadured photometric distributions.
    """

    def __init__(self, ens, zgrid=None):
        self._prior_base(ens)

    def _prior_base(self, ens, zgrid=None):
        if type(ens) is qp.ensemble.Ensemble:
            z_edges = ens.metadata()["bins"][0]
            z = 0.5 * (z_edges[1:] + z_edges[:-1])
            nzs = ens.objdata()["pdfs"]
        elif type(ens) is list:
            z = ens[0]
            nzs = ens[1]
        else:
            raise ValueError("Invalid ensemble type=={}".format(type(ens)))

        if zgrid is not None:
            nzs = [np.interp(zgrid, z, nz) for nz in nzs]
            self.z = zgrid
        else:
            self.z = z

        self.nzs = self._normalize(nzs)
        self.nz_mean = np.mean(self.nzs, axis=0)
        self.nz_cov = np.cov(self.nzs, rowvar=False)
        self.prior_mean = None
        self.prior_cov = None
        self.prior_chol = None

    def _normalize(self, nzs):
        norms = np.sum(nzs, axis=1)
        nzs = nzs / norms[:, None]
        return nzs

    def get_prior(self):
        """
        Returns the calibrated prior distribution for the model
        parameters given the measured photometric distributions.
        """
        if (self.prior_mean is None) | (self.prior_cov is None):
            self.prior = self._get_prior()
        return self.prior_mean, self.prior_cov, self.prior_chol

    def _get_prior(self):
        raise NotImplementedError

    def sample_prior(self):
        """
        Draws a sample from the prior distribution.
        """
        prior_mean, prior_cov, prior_chol = self.get_prior()
        prior_dist = mvn(np.zeros_like(prior_mean), np.ones_like(prior_mean))
        alpha = prior_dist.rvs()
        if type(alpha) is np.float64:
            alpha = np.array([alpha])
        values = prior_mean + prior_chol @ alpha
        param_names = self._get_params_names()
        samples = {param_names[i]: values[i] for i in range(len(values))}
        return samples

    def save_prior(self, path="./"):
        """
        Saves the prior distribution to a file.
        """
        prior_mean, prior_cov = self.get_prior()
        np.save(path + "prior_mean.npy", prior_mean)
        np.save(path + "prior_cov.npy", prior_cov)

    def test_prior(self):
        """
        Tests the distribution of parameters is
        actually Gaussian.
        """
        params = self.params
        shape = params.shape
        if len(shape) == 3:
            # For the sacc prior
            n, m, k = shape
            params = np.reshape(params, (n * m, k))
        params = np.real(params)

        prior_mean, prior_cov, _ = self.get_prior()
        prior_std = np.sqrt(np.diag(prior_cov))

        p_values = []
        for i, param in enumerate(params):
            result = kstest(param, "norm", args=(prior_mean[i], prior_std[i]))
            p_value = result.pvalue
            param_name = self.params_names[i]
            if result.pvalue < 0.05:
                print(
                    "Warning: p-value for {} being Gaussianly distributed is {}".format(
                        param_name, p_value
                    )
                )

            p_values.append(p_value)
        return p_values

    def plot_prior(self, mode="1D", **kwargs):
        names = self.params_names
        params = self.params
        shape = params.shape
        if len(shape) == 3:
            # For the sacc prior
            n, m, k = shape
            params = np.reshape(params, (n * m, k))
        params = params.T
        params = np.real(params)
        chain = MCSamples(
            samples=params,
            names=names,
            settings={
                "mult_bias_correction_order": 0,
                "smooth_scale_2D": 0.4,
                "smooth_scale_1D": 0.3,
            },
        )
        g = plots.getSubplotPlotter(subplot_size=2.5)
        g.settings.axes_fontsize = 20
        g.settings.legend_fontsize = 20
        g.settings.axes_labelsize = 20
        if mode == "2D":
            g.triangle_plot([chain], filled=True)
        elif mode == "1D":
            g.plots_1d([chain], **kwargs)
