import numpy as np
import sacc
from numpy.linalg import cholesky
from scipy.linalg import block_diag
from .prior_base import PriorBase
from .prior_shifts import PriorShifts
from .prior_shifts_widths import PriorShiftsWidths
from .prior_comb import PriorComb
from .prior_gp import PriorGP
from .prior_pca import PriorPCA
from .utils import make_cov_posdef


class PriorSacc(PriorBase):
    def __init__(self, sacc_file, model="Shifts", compute_crosscorrs="Full", **kwargs):
        if model == "Shifts":
            self.model = PriorShifts
            self.sacc_tracer = sacc.NZShiftUncertainty
        if model == "ShiftsWidths":
            self.model = PriorShiftsWidths
            self.sacc_tracer = sacc.NZShiftStretchUncertainty
        if model == "GP":
            self.model = PriorGP
            self.sacc_tracer = sacc.NZLinearUncertainty
        if model == "Comb":
            self.model = PriorComb
            self.sacc_tracer = sacc.NZLinearUncertainty
        if model == "PCA":
            self.model = PriorPCA
            self.sacc_tracer = sacc.NZLinearUncertainty
        self.sacc_file = sacc_file.copy()
        self.compute_crosscorrs = compute_crosscorrs
        self.tracers = sacc_file.tracers
        self.model_objs = self._make_model_objects(**kwargs)
        self.params = None
        self.params_names = None
        self.prior_mean = None
        self.prior_cov = None
        self.prior_chol = None
        self.prior_transform = None

    def save2sacc(self, name):
        self._get_prior()
        tracer = self.sacc_tracer(
            name,
            list(self.model_objs.keys()),
            self.prior_mean,
            self.prior_transform.T,
        )
        self.sacc_file.add_tracer_uncertainty_object(tracer)
        return self.sacc_file

    def _make_model_objects(self, **kwargs):
        model_objs = {}
        for tracer_name in list(self.tracers.keys()):
            print("Making model for ", tracer_name)
            tracer = self.tracers[tracer_name]
            ens = tracer.ensemble
            model_obj = self.model(ens, **kwargs)
            model_objs[tracer_name] = model_obj
        return model_objs

    def _get_prior(self):
        self.get_params()
        self.get_params_names()
        self.prior_mean = np.array(
            [np.mean(param_sets, axis=1) for param_sets in self.params]
        ).flatten()
        if self.compute_crosscorrs == "Full":
            print("Computing full covariance matrix")
            params = []
            for param_sets in self.params:
                for param_set in param_sets:
                    params.append(param_set)
            params = np.array(params)
            cov = np.cov(params)
            cov = make_cov_posdef(cov)
            chol = cholesky(cov)
            transform = chol
        elif self.compute_crosscorrs == "BinWise":
            covs = []
            chols = []
            transforms = []
            for tracer_name in list(self.tracers.keys()):
                model_obj = self.model_objs[tracer_name]
                _, cov, chol = model_obj.get_prior()
                transform = model_obj.get_transform()
                covs.append(cov)
                chols.append(chol)
                transforms.append(transform)
            covs = np.array(covs)
            chols = np.array(chols)
            cov = block_diag(*covs)
            chol = block_diag(*chols)
            transform = block_diag(*transforms)
        elif self.compute_crosscorrs == "None":
            stds = []
            Ws = []
            for param_sets in self.params:
                for param_set in param_sets:
                    stds.append(np.std(param_set))
            for tracer_name in list(self.tracers.keys()):
                model_obj = self.model_objs[tracer_name]
                _, cov, chol = model_obj.get_prior()
                inv_chol = np.linalg.pinv(chol)
                transform = model_obj.get_transform()
                W = transform @ inv_chol
                Ws.append(W)
            W = block_diag(*Ws)
            stds = np.array(stds)
            cov = np.diag(stds**2)
            chol = np.diag(stds)
            transform = W @ chol
        else:
            raise ValueError(
                "Invalid compute_crosscorrs=={}".format(self.compute_crosscorrs)
            )
        self.prior_cov = cov
        self.prior_chol = chol
        self.prior_transform = transform

    def _get_params_names(self):
        params_names = []
        for tracer_name in list(self.tracers.keys()):
            model_obj = self.model_objs[tracer_name]
            params_names_set = model_obj._get_params_names()
            for param_name in params_names_set:
                param_name = tracer_name + "__" + param_name
                params_names.append(param_name)
        return np.array(params_names)

    def _get_params(self):
        params = []
        for tracer_name in list(self.tracers.keys()):
            model_obj = self.model_objs[tracer_name]
            params_sets = model_obj._get_params()
            params.append(params_sets)
        try:
            np.array(params)
        except:
            raise ValueError("Each QP ensemble has different number of realizations")
        return np.array(params)
