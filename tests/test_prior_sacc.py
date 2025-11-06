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
    ss = prior.save2sacc(file_name="./test.sacc", tracer_name=name)
    ss_loaded = sacc.Sacc.load_fits("./test.sacc")

    assert (ss.tracer_uncertainties[name].tracer_names ==
            ss_loaded.tracer_uncertainties[name].tracer_names)
    assert (ss.tracer_uncertainties[name].mean ==
            ss_loaded.tracer_uncertainties[name].mean).all()
    assert ss.tracer_uncertainties[name].nparams == ss_loaded.tracer_uncertainties[name].nparams
