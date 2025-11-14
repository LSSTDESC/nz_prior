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
    z = prior.z
    nzs = prior.nzs
    m, n = nzs.shape
    (k,) = z.shape
    # check of ensamble dimensions
    assert n == k
    # check normalization
    dz = z[1] - z[0]
    norms = np.sum(nzs, axis=1) * dz
    mean_norm = np.mean(norms)
    assert np.abs(mean_norm - 1.0) < 1e-3

