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
    return nzp.PriorShiftsWidths(ens)


def test_prior():
    prior = make_prior()
    prior = prior.get_prior()
    assert prior is not None


def test_sample_prior():
    prior = make_prior()
    prior_sample = prior.sample_prior()
    assert len(prior_sample) == 2


def test_model():
    model = nzp.shift_and_width_model
    prior = make_prior()
    shift = 0.1 #rior_sample["delta_z"]
    width = 1.1 #prior_sample["width_z"]
    mu = np.average(prior.z, weights=prior.nz_mean)
    std = np.sqrt(np.average((prior.z - mu) ** 2, weights=prior.nz_mean))

    new_z = prior.z + mu
    new_nz = model(prior.z, prior.nz_mean, shift, 1)
    _shift = np.average(new_z, weights=prior.nz_mean) - mu
    assert np.isclose(_shift, shift, atol=1e-2)

    new_nz = model(prior.z, prior.nz_mean, shift, width)
    new_std = np.sqrt(np.average((prior.z - mu) ** 2, weights=new_nz))
    _width = new_std / std
    print(_shift, shift)
    print(_width, width)
    assert np.isclose(_shift, shift, atol=1e-2)
    assert np.isclose(_width, width, atol=1e-2)
