import numpy as np
from numpy.linalg import eig
from .prior_linear import PriorLinear


class PriorPCA(PriorLinear):
    """
    Prior for the PCA model.
    """

    def __init__(self, ens, n=5, zgrid=None):
        super().__init__(ens, n=n, zgrid=zgrid)

    def _find_funcs(self):
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
