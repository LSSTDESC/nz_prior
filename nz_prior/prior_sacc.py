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
from .prior_linear import PriorLinear
from .utils import make_cov_posdef


class PriorSacc(PriorBase):
    def __init__(self, sacc_file, model_name="Shifts", compute_crosscorrs="Full", **kwargs):
        self.model_name = model_name
        if model_name == "Shifts":
            self.model = PriorShifts
            self.sacc_tracer = sacc.NZShiftUncertainty
        elif model_name == "ShiftsWidths":
            self.model = PriorShiftsWidths
            self.sacc_tracer = sacc.NZShiftStretchUncertainty
        elif model_name == "GP":
            self.model = PriorGP
            self.sacc_tracer = sacc.NZLinearUncertainty
        elif model_name == "Comb":
            self.model = PriorComb
            self.sacc_tracer = sacc.NZLinearUncertainty
        elif model_name == "PCA":
            self.model = PriorPCA
            self.sacc_tracer = sacc.NZLinearUncertainty
        else:
            raise ValueError("Model not implemented =={}".format(model_name))
        self.sacc_file = sacc_file.copy()
        self.compute_crosscorrs = compute_crosscorrs
        self.tracers = sacc_file.tracers
        kwargs.pop("model_name", None)
        kwargs.pop("compute_crosscorrs", None)
        self.model_objs = self._make_model_objects(**kwargs)
        self.params = None
        self.params_names = None
        self.prior_mean = None
        self.prior_cov = None
        self.prior_chol = None
        # Only for linear models
        self.prior_model = None

    def save(self, tracer_name=None):
        # Compute the prior if not already done
        self.get_prior()

        # if model is instance of PriorLinear 
        # add the model matrix to the mean and chol
        mean = self.prior_mean
        chol = self.prior_chol
        if issubclass(self.model, PriorLinear):
            self.prior_model = []
            for tracer in self.model_objs.values():
                self.prior_model.append(tracer.get_funcs())
            self.prior_model = block_diag(*self.prior_model)
            mean = self.prior_model @ self.prior_mean
            chol = self.prior_model @ self.prior_chol
        if tracer_name is None:
            tracer_name = self.model_name
        tracer = self.sacc_tracer(
            tracer_name,
            list(self.model_objs.keys()),
            mean,
            chol,
        )
        # Add the tracer uncertainty object to the sacc file
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
        # The mean computation is the same for all cross-corr options
        means = []
        for model_obj in self.model_objs.values():
            model_obj._get_prior()
            means.append(model_obj.prior_mean)
        self.prior_mean = np.array(means).flatten()
        self.nparams = np.sum([model_obj.nparams for model_obj in self.model_objs.values()])

        # Now compute the covariance and transform based on cross-corr option
        self.get_params()
        self.get_params_names()
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
        elif self.compute_crosscorrs == "BinWise":
            covs = []
            chols = []
            for tracer_name in list(self.tracers.keys()):
                model_obj = self.model_objs[tracer_name]
                model_obj._get_prior()
                cov = model_obj.prior_cov
                chol = model_obj.prior_chol
                covs.append(cov)
                chols.append(chol)
            covs = np.array(covs)
            chols = np.array(chols)
            cov = block_diag(*covs)
            chol = block_diag(*chols)
        elif self.compute_crosscorrs == "None":
            covs = []
            chols = []
            for tracer_name in list(self.tracers.keys()):
                model_obj = self.model_objs[tracer_name]
                model_obj._get_prior()
                cov = model_obj.prior_cov
                chol = model_obj.prior_chol
                covs.append(cov)
                chols.append(chol)
            covs = np.array(covs)
            chols = np.array(chols)
            diag_covs = [np.diag(np.diag(cov)) for cov in covs]
            diag_chols = [cholesky(cov) for cov in diag_covs]
            cov = block_diag(*diag_covs)
            chol = block_diag(*diag_chols)
        else:
            raise ValueError(
                "Invalid compute_crosscorrs=={}".format(self.compute_crosscorrs)
            )
        self.prior_cov = cov
        self.prior_chol = chol

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
