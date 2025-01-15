import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import norm
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorFourier(PriorBase):
    """
    Prior for the Fourier model.
    """

    def __init__(self, ens, n=10, zgrid=None):
        self._prior_base(ens, zgrid=zgrid)
        self.n = n
        self._find_prior()
        self.params_names = self._get_params_names()
        self.params = self._get_params()
        self.test_prior()

    def _find_prior(self):
        self.Ws = self._find_weights()

    def _find_weights(self):
        Ws = []
        for nz in self.nzs:
            d_nz = nz - self.nz_mean
            W = np.fft.fft(d_nz)
            W = W[: self.n]
            Ws.append(W)
        return np.array(Ws)

    def _get_prior(self):
        mean = np.mean(self.Ws, axis=0)
        cov = np.cov(self.Ws.T)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        self.prior_mean = np.real(mean)
        self.prior_cov = np.real(cov)
        self.prior_chol = np.real(chol)

    def _get_params(self):
        return self.Ws.T

    def _get_params_names(self):
        return ["W_{}".format(i) for i in range(len(self.Ws.T))]
