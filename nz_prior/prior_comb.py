import numpy as np
from scipy.stats import norm
from numpy.linalg import eig
from .prior_linear import PriorLinear


class PriorComb(PriorLinear):
    """
    Prior for the comb model.
    """

    def __init__(self, ens, n=5, zgrid=None):
        super().__init__(ens, n=n, zgrid=zgrid)

    def _get_funcs(self):
        zmax = np.max(self.z)
        zmin = np.min(self.z)
        dz = (zmax - zmin) / self.n
        zmeans = [(zmin + dz / 2) + i * dz for i in range(self.n)]
        combs = [norm(zmeans[i], dz / 2) for i in np.arange(self.n)]
        return combs

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
