import numpy as np
from numpy.linalg import eig, cholesky
from scipy.stats import multivariate_normal as mvn
from .prior_base import PriorBase
from .utils import make_cov_posdef


class PriorGP(PriorBase):
    """
    Prior for the moments model.
    """

    def __init__(self, ens, n=None, zgrid=None):
        self.n = n
        super().__init__(ens, zgrid=zgrid)

    def _find_q(self):
        z_edges = self.ens.metadata()["bins"][0]
        z = 0.5 * (z_edges[1:] + z_edges[:-1])
        q_edges = np.linspace(z[0], z[-1], self.n+1)
        q = 0.5 * (q_edges[1:] + q_edges[:-1])
        return q

    def _find_dnzs(self):
        nzs = []
        nqs = []
        nzqs = []
        self.q = self._find_q()
        for nz in self.nzs:
            nq = np.interp(self.q, self.z, nz)
            nzq = np.append(nz, nq)
            nqs.append(nq)
            nzs.append(nz)
            nzqs.append(nzq)
        nqs = np.array(nqs)
        nzs = np.array(nzs)
        nzqs = np.array(nzqs)
        return nzs, nqs, nzqs

    def _compute_prior_samples(self):
        self.nzs, self.nqs, self.nzqs = self._find_dnzs()
        self.nq_mean = np.mean(self.nqs, axis=0)
        self.nz_mean = np.mean(self.nzs, axis=0)
        self.nzq_mean = np.mean(self.nzqs, axis=0)
        dnzqs = self.nzqs - self.nzq_mean
        cov_zzqq = np.cov(dnzqs.T)
        cov_qq = cov_zzqq[len(self.nz_mean):, len(self.nz_mean):]
        cov_zq = cov_zzqq[:len(self.nz_mean), len(self.nz_mean):]
        inv_cov_qq = np.linalg.pinv(cov_qq)
        self.W = np.dot(cov_zq, inv_cov_qq)

    def _get_prior(self):
        self._compute_prior_samples()
        self.prior_mean = self.nq_mean
        d_cov = np.cov(self.nqs, rowvar=False)
        self.prior_cov = make_cov_posdef(d_cov)
        self.prior_chol = cholesky(self.prior_cov)

    def get_params(self):
        return self.nqs.T

    def get_params_names(self):
        return ["gp_{}".format(i) for i in range(len(self.nqs.T))]
