import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import norm
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorPCA(PriorBase):
    """
    Prior for the PCA model.
    """
    def __init__(self, ens, npca=5, zgrid=None):
        self._prior_base(ens, zgrid=zgrid)
        self.npca = npca
        d_nzs = self.nzs - self.nz_mean
        d_cov = np.cov(d_nzs, rowvar=False)
        self.eigvals, self.eigvecs = eig(d_cov)
        self.eigvecs = np.real(self.eigvecs)
        self.eigvals = np.real(self.eigvals)
        idx = np.argsort(self.eigvals)[::-1]
        self.eigvals = self.eigvals[idx]
        self.eigvecs = self.eigvecs[:, :npca]
        self.eigvals = self.eigvals[:npca]
        self.eigvecs = self.eigvecs.T
        self._find_prior()
        self.params = self._get_params()
        self.params_names = self._get_params_names()
        self.test_prior()

    def _find_prior(self):
        self.Ws = self._find_weights()

    def _find_weights(self):
        Ws = []
        for nz in self.nzs:
            W = [np.dot(nz, self.eigvecs[i]) for i in np.arange(self.npca)]
            Ws.append(W)
        return np.array(Ws)

    def _get_prior(self):
        mean = np.mean(self.Ws, axis=0)
        cov = np.cov(self.Ws.T)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def _get_params(self):
        return self.Ws.T

    def _get_params_names(self):
        return ['W_{}'.format(i) for i in range(len(self.Ws.T))]
