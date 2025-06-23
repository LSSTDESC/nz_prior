import numpy as np
from numpy.linalg import eig
from .prior_linear import PriorLinear


class PriorPCA(PriorLinear):
    """
    Prior for the PCA model.
    """

    def __init__(self, ens, n=5, zgrid=None):
        super().__init__(ens, n=n, zgrid=zgrid)
        self.funcs = self._get_funcs()
        self.Ws = self._get_weights()

    def _get_weights(self):
        Ws = []
        for nz in self.nzs:
            dnz = nz - self.nz_mean
            W = [np.dot(dnz, self.funcs.T[i]) for i in np.arange(self.n)]
            Ws.append(W)
        return np.array(Ws)

    def _get_funcs(self):
        d_nzs = self.nzs - self.nz_mean
        d_cov = np.cov(d_nzs, rowvar=False)
        eigvals, eigvecs = eig(d_cov)
        eigvecs = np.real(eigvecs)
        eigvals = np.real(eigvals)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvecs = eigvecs[:, : self.n]
        eigvals = eigvals[: self.n]
        return eigvecs
