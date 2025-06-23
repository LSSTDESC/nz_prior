import numpy as np
from .prior_linear import PriorLinear


class PriorGP(PriorLinear):
    """
    Prior for the moments model.
    """

    def __init__(self, ens, n=5, nz_fid=None):
        super().__init__(ens, n=n, nz_fid=nz_fid)
        self.Ws = self._get_weights()
        self.W_sys = self._get_sys_weights()
        self.funcs = self._get_funcs()

    def _find_q(self):
        z_edges = self.ens.metadata()["bins"][0]
        z = 0.5 * (z_edges[1:] + z_edges[:-1])
        q_edges = np.linspace(z[0], z[-1], self.n + 1)
        q = 0.5 * (q_edges[1:] + q_edges[:-1])
        return q

    def _get_weights(self):
        self.q = self._find_q()
        Ws = [np.interp(self.q, self.z, nz) for nz in self.nzs]
        return np.array(Ws)

    def _get_sys_weights(self):
        self.q = self._find_q()
        return np.interp(self.q, self.z, self.nz_fid)

    def _get_funcs(self):
        n1, m1 = self.Ws.shape
        n2, m2 = self.nzs.shape
        nzqs = np.zeros((n1, m2 + m1))
        nzqs[:, :m2] = self.nzs
        nzqs[:, m2:] = self.Ws
        nzq_mean = np.mean(nzqs, axis=0)
        dnzqs = nzqs - nzq_mean
        cov_zzqq = np.cov(dnzqs.T)
        cov_qq = cov_zzqq[len(self.nz_mean) :, len(self.nz_mean) :]
        cov_zq = cov_zzqq[: len(self.nz_mean), len(self.nz_mean) :]
        inv_cov_qq = np.linalg.pinv(cov_qq)
        wiener = np.dot(cov_zq, inv_cov_qq)
        return wiener
