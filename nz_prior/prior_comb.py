import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import norm
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorComb(PriorBase):
    """
    Prior for the comb model.
    """
    def __init__(self, ens, ncombs=10, zgrid=None):
        self._prior_base(ens, zgrid=zgrid)
        self.ncombs = ncombs
        zmax = np.max(self.z)
        zmin = np.min(self.z)
        dz = (zmax - zmin)/ncombs
        zmeans = [(zmin + dz/2) + i*dz for i in range(ncombs)]
        self.combs = {}
        for i in np.arange(self.ncombs):
            self.combs[i] = norm(zmeans[i], dz/2)
        self._find_prior()

    def _find_prior(self):
        self.Ws = self._find_weights()

    def _find_weights(self):
        Ws = []
        for nz in self.nzs:
            W = [np.dot(nz, self.combs[i].pdf(self.z)) for i in np.arange(self.ncombs)]
            Ws.append(W/np.sum(W))
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
