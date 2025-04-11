import numpy as np
from numpy.linalg import cholesky
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorLinear(PriorBase):
    """
    Prior for the PCA model.
    """

    def __init__(self, ens, n=5, zgrid=None):
        self.n = n
        super().__init__(ens, zgrid=zgrid)

    def _compute_prior_samples(self):
        self.funcs = self._find_funcs()
        self.Ws = self._find_weights()

    def _find_funcs(self):
        raise NotImplementedError

    def _find_weights(self):
        Ws = []
        for nz in self.nzs:
            dnz = nz - self.nz_mean
            W = [np.dot(dnz, self.funcs.T[i]) for i in np.arange(self.n)]
            Ws.append(W)
        return np.array(Ws)

    def _get_prior(self):
        self._compute_prior_samples()
        mean = np.mean(self.Ws, axis=0)
        cov = np.cov(self.Ws.T)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def get_params(self):
        return self.Ws.T

    def get_params_names(self):
        return ["W_{}".format(i) for i in range(len(self.Ws.T))]
