import qp
import numpy as np
import nz_prior as nzp


def make_qp_ens(file):
    zs = file["zs"]
    nzs = file["pzs"]
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz / 2, zs[-1] + dz / 2)
    q = qp.Ensemble(qp.hist, data={"bins": zs_edges, "pdfs": nzs})
    return q


def make_prior(nparams=10):
    file = np.load("tests/dummy.npz")
    ens = make_qp_ens(file)
    return nzp.PriorComb(ens, nparams=nparams)


def test_prior():
    prior = make_prior()
    prior = prior.get_prior()
    assert prior is not None


def test_sample_prior():
    nparams = 10
    prior = make_prior(nparams=nparams)
    prior_sample = prior.sample_prior()
    assert len(list(prior_sample.values())) == nparams
