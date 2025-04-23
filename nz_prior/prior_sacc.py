import numpy as np
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
        if model == "ShiftsWidths":
            self.model = PriorShiftsWidths
        if model == "GP":
            self.model = PriorGP
        if model == "Comb":
            self.model = PriorComb
        if model == "PCA":
            self.model = PriorPCA

        self.compute_crosscorrs = compute_crosscorrs
        self.tracers = sacc_file.tracers
        self.model_objs = self._make_model_objects(**kwargs)
        self.params = None
        self.params_names = None
        self.prior_mean = None
        self.prior_cov = None
        self.prior_chol = None

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
        self.prior_mean = np.array(
            [np.mean(param_sets, axis=1) for param_sets in self.params]
        ).flatten()
        if self.compute_crosscorrs == "Full":
            params = []
            for param_sets in self.params:
                for param_set in param_sets:
                    params.append(param_set)
            params = np.array(params)
            cov = np.cov(params)
        elif self.compute_crosscorrs == "BinWise":
            covs = []
            for p in self.params:
                covs.append(np.cov(p))
            covs = np.array(covs)
            cov = block_diag(*covs)
        elif self.compute_crosscorrs == "None":
            stds = []
            for param_sets in self.params:
                for param_set in param_sets:
                    stds.append(np.std(param_set))
            stds = np.array(stds)
            cov = np.diag(stds**2)
        else:
            raise ValueError(
                "Invalid compute_crosscorrs=={}".format(self.compute_crosscorrs)
            )
        self.prior_cov = make_cov_posdef(cov)
        self.prior_chol = cholesky(self.prior_cov)

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
