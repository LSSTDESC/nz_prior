import numpy as np
from scipy.stats import norm
from .prior_linear import PriorLinear


class PriorComb(PriorLinear):
    """
    Prior for the comb model.
    """

    def __init__(self, ens, n=5, zgrid=None):
        super().__init__(ens, n=n, zgrid=zgrid)
        self.funcs = self._get_funcs()
        self.Ws = self._get_weights()

    def _get_funcs(self):
        zmax = np.max(self.z)
        zmin = np.min(self.z)
        dz = (zmax - zmin) / self.n
        zmeans = [(zmin + dz / 2) + i * dz for i in range(self.n)]
        combs = [norm(zmeans[i], dz / 2) for i in np.arange(self.n)]
        combs = np.array([comb.pdf(self.z) for comb in combs])
        return combs

    def _get_weights(self):
        Ws = []
        for nz in self.nzs:
            dnz = nz - self.nz_mean
            W = [np.dot(dnz, self.funcs.T[i]) for i in np.arange(self.n)]
            Ws.append(W)
        return np.array(Ws)
