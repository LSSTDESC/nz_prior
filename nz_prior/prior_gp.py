import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import multivariate_normal as mvn
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorGP(PriorBase):
    """
    Prior for the moments model.
    The moments model assumes that meausred photometric distribution
    is Gaussian meaning that it can be fully described by its mean and
    covariance matrix. Conceptually, this is equavalent to a 
    Gaussian process regressio for a given p(z). The details can be found 
    in the paper: 2301.11978

    Some measured photometric distributions will possess non-invertible
    covariance matrices. If this is the case, PriorMoments will
    attempt regularize the covariance matrix by adding a small jitter
    to its eigen-values. If this fails, the covariance matrix will be
    diagonalized.
    """
    def __init__(self, ens, zgrid=None):
        self._prior_base(ens, zgrid=zgrid)

    def _get_prior(self):
        self.prior_mean = self.nz_mean
        self.prior_cov = make_cov_posdef(self.nz_cov)
        self.prior_chol = cholesky(self.prior_cov)

    def _get_params(self):
        return self.nzs.T

    def _get_params_names(self):
        return ['nz_{}'.format(i) for i in range(len(self.nzs.T))]
