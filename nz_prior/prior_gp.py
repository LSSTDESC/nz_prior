import numpy as np
from numpy.linalg import cholesky
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

    def _compute_prior_samples(self):
        self.Ws = self._find_weights()
        self.funcs = self._find_funcs()

    def _find_weights(self):
        self.q = self._find_q()
        Ws = [np.interp(self.q, self.z, nz) for nz in self.nzs]
        return np.array(Ws)

    def _find_funcs(self):
        n1, m1 = self.Ws.shape
        n2, m2 = self.nzs.shape
        nzqs = np.zeros((n1, m2+m1))
        nzqs[:, :m2] = self.nzs
        nzqs[:, m2:] = self.Ws
        nzq_mean = np.mean(nzqs, axis=0)
        dnzqs = nzqs - nzq_mean
        cov_zzqq = np.cov(dnzqs.T)
        cov_qq = cov_zzqq[len(self.nz_mean):, len(self.nz_mean):]
        cov_zq = cov_zzqq[:len(self.nz_mean), len(self.nz_mean):]
        inv_cov_qq = np.linalg.pinv(cov_qq)
        wiener = np.dot(cov_zq, inv_cov_qq)
        return wiener

    def _get_prior(self):
        self._compute_prior_samples()
        mean = np.mean(self.Ws, axis=0)
        d_cov = np.cov(self.Ws, rowvar=False)
        cov = make_cov_posdef(d_cov)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def get_params(self):
        return self.Ws.T

    def get_params_names(self):
        return ["gp_{}".format(i) for i in range(len(self.Ws.T))]
