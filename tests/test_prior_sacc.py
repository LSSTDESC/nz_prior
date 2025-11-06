import qp
import sacc
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
    zs = file["zs"]
    nzs = file["pzs"]
    dz = np.mean(np.diff(zs))
    zs_edges = np.append(zs - dz / 2, zs[-1] + dz / 2)
    ens = qp.Ensemble(qp.hist, data={"bins": zs_edges, "pdfs": nzs})
    s = sacc.Sacc()
    s.add_tracer("QPNZ", "source_0", ens, z=zs)
    s.add_tracer("QPNZ", "source_1", ens, z=zs)
    return nzp.PriorSacc(s, model_name="PCA", compute_crosscorrs="BinWise", nparams=5)


def test_prior():
    prior = make_prior()
    prior = prior.get_prior()
    assert prior is not None


def test_sample_prior():
    prior = make_prior()
    prior_sample = prior.sample_prior()
    prior_params = len(list(prior_sample.values()))
    assert prior_params == 10


def test_save_prior():
    prior = make_prior()
    name = "test"
    ss = prior.save(tracer_name=name)

    assert (ss.tracer_uncertainties[name].tracer_names ==
            list(prior.tracers.keys()))
    assert (ss.tracer_uncertainties[name].mean ==
            prior.prior_mean).all()
