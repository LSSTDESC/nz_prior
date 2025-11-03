import numpy as np
from getdist import plots, MCSamples
from scipy.stats import multivariate_normal as mvn
from scipy.stats import kstest
from .utils import normalize
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
        """
        Initializes the prior class.
        Parameters
        ----------
        ens : qp.ensemble.Ensemble or list
            Ensemble of measured photometric distributions.
        zgrid : array_like, optional
            Redshift grid to use for the prior. If None, the redshift
            grid of the ensemble is used.
        """
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

        self.ens = ens
        self.nzs = normalize(nzs)
        self.nz_mean = np.mean(self.nzs, axis=0)
        self.nz_cov = np.cov(self.nzs, rowvar=False)
        self.params = None
        self.params_names = None
        self.prior_mean = None
        self.prior_cov = None
        self.prior_chol = None
        self.prior_transform = None
        self.prior_basis = None # Only for linear models

    def get_prior(self):
        """
        Returns the calibrated prior distribution for the model
        parameters given the measured photometric distributions.
        """
        if (self.prior_mean is None) | (self.prior_cov is None):
            self.prior = self._get_prior()
        return self.prior_mean, self.prior_cov, self.prior_chol

    def get_transform(self):
        """
        Returns the transformation matrix for the model parameters.
        """
        if self.prior_transform is None:
            self._get_prior()
        return self.prior_transform

    def get_params(self):
        """
        Returns the parameters of the model.
        """
        if self.params is None:
            self.params = self._get_params()
        return self.params

    def get_params_names(self):
        """
        Returns the names of the parameters of the model.
        """
        if self.params_names is None:
            self.params_names = self._get_params_names()
        return self.params_names

    def _get_prior(self):
        raise NotImplementedError

    def _get_params(self):
        raise NotImplementedError

    def _get_params_names(self):
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
        param_names = self.get_params_names()
        samples = {param_names[i]: values[i] for i in range(len(values))}
        return samples

    def plot_prior(
        self,
        order=None,
        labels=None,
        mode="1D",
        add_prior=True,
        **kwargs,
    ):
        params = self.get_params()
        names = self.get_params_names()
        if labels is None:
            labels = names
        shape = params.shape
        if len(shape) == 3:
            # For the sacc prior
            n, m, k = shape
            params = np.reshape(params, (n * m, k))
        if order is not None:
            print("Order: ", order)
            params = params[order]
            names = names[order]
            labels = labels[order]
        params = params.T
        params = np.real(params)
        chain = MCSamples(
            samples=params,
            names=names,
            labels=labels,
            label="Measured Distribution",
            settings={
                "mult_bias_correction_order": 0,
                "smooth_scale_2D": 0.4,
                "smooth_scale_1D": 0.3,
            },
        )
        g = plots.getSubplotPlotter(subplot_size=1.5)
        g.settings.axes_fontsize = 20
        g.settings.legend_fontsize = 20
        g.settings.axes_labelsize = 20
        chains = [chain]
        if add_prior:
            samples = []
            for i in range(2000):
                sample = self.sample_prior()
                _sample = np.array([s for s in sample.values()])
                samples.append(_sample)
            samples = np.array(samples)
            if order is not None:
                samples = samples[:, order]
            prior_chain = MCSamples(
                samples=samples,
                names=names,
                label="Gaussian Prior",
                settings={
                    "mult_bias_correction_order": 0,
                    "smooth_scale_2D": 0.4,
                    "smooth_scale_1D": 0.3,
                },
            )
            chains.append(prior_chain)
        if mode == "2D":
            g.triangle_plot(chains, filled=True, **kwargs)
        elif mode == "1D":
            g.plots_1d(chains, share_y=True, **kwargs)
        return g
