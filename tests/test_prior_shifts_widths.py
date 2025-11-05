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
    shift = 0.1
    width = 1.1
    model = nzp.shift_and_width_model
    mu = 0.5
    std = 0.1
    z = np.linspace(mu - 5*std, mu + 5*std, 100)
    nz = np.exp(-0.5 * ((z - mu) / std) ** 2)
    nz /= np.sum(nz)

    new_nz = model(z, nz, shift, 1)
    _mu = np.average(z, weights=new_nz)
    _shift = _mu - mu
    print(_mu)
    assert np.isclose(_shift, shift, atol=1e-3)

    new_nz = model(z, nz, shift, width)
    _mu = np.average(z, weights=new_nz)
    _shift = _mu - mu
    _std = np.sqrt(np.average((z - _mu) ** 2, weights=new_nz))
    _width = _std / std
    assert np.isclose(_shift, shift, atol=1e-3)
    assert np.isclose(_width, width, atol=1e-3)
