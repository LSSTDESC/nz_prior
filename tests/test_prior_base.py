import qp
import numpy as np
import nz_prior as nz


def make_qp_ens(file):
    zs = file["zs"]
    nzs = file["pzs"]
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz / 2, zs[-1] + dz / 2)
    q = qp.Ensemble(qp.hist, data={"bins": zs_edges, "pdfs": nzs})
    return q


def test_base():
    file = np.load("tests/dummy.npz")
    ens = make_qp_ens(file)
    prior = nz.PriorBase(ens)
    m, n = prior.nzs.shape
    (k,) = prior.z.shape
    nzs = file["pzs"]
    nzs /= np.sum(nzs, axis=1)[:, None]
    assert n == k
    assert np.allclose(prior.nz_mean, np.mean(nzs, axis=0))
