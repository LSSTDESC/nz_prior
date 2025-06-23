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
        self.funcs = None
        self.Ws = None

    def get_funcs(self):
        if self.funcs is None:
            self.funcs = self._get_funcs()
        return self.funcs

    def get_weights(self):
        if self.Ws is None:
            self.Ws = self._get_weights()
        return self.Ws

    def _get_funcs(self):
        raise NotImplementedError

    def _get_weights(self):
        raise NotImplementedError

    def _get_prior(self):
        self.get_funcs()
        self.get_weights()
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
        return ["W_{}".format(i) for i in range(len(self.Ws.T))]
