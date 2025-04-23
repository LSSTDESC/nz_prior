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


def make_prior():
    file = np.load("tests/dummy.npz")
    ens = make_qp_ens(file)
    return nzp.PriorShifts(ens)


def test_prior():
    prior = make_prior()
    prior = prior.get_prior()
    assert prior is not None


def test_sample_prior():
    prior = make_prior()
    prior_sample = prior.sample_prior()
    prior_params = len(list(prior_sample.values()))
    assert prior_params == 1


def test_model():
    model = nzp.shift_and_width_model
    prior = make_prior()
    shift = prior.sample_prior()["delta_z"]
    output = model(prior.z, prior.nz_mean, shift, 0)
    assert (prior.z == output[0]).all()
    assert len(output[1]) == len(prior.nz_mean)
