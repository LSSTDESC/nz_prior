import qp
import numpy as np
import nz_prior as nzp

def make_qp_ens(file):
    zs = file['zs']
    nzs = file['pzs']
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz/2, zs[-1] + dz/2)
    q = qp.Ensemble(qp.hist, data={"bins":zs_edges, "pdfs":nzs})
    return q


def make_prior():
    file = np.load('tests/dummy.npz')
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
    prior_sample = prior.sample_prior()
    shift = prior_sample['delta_z']
    width = prior_sample['width_z']
    input = np.array([prior.z, prior.nz_mean])
    output = model(input, [shift, width])
    assert (prior.z == output[0]).all()
    assert len(output[1]) == len(prior.nz_mean)
