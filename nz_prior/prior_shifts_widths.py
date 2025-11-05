import numpy as np
from numpy.linalg import cholesky
from .prior_base import PriorBase
from .utils import make_cov_posdef
from .models import shift_and_width_model


class PriorShiftsWidths(PriorBase):
    """
    Prior for the shifts and widths model.
    The shifts and widths model assumes that the variation in the measured
    photometric distributions can be captured by varying the mean and the
    standard deviation of a fiducial n(z) distribution.

    The calibration method was written by Tilman Tröster.
    The shift prior is given by a Gaussian distribution with zero mean
    standard deviation the standard deviation in the mean of
    the measured photometric distributions.
    The width is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The width prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    This is similar to how the shift prior is calibrated in the shift model.
    """

    def __init__(self, ens, zgrid=None, optimize_widths=None):
        super().__init__(ens, zgrid=zgrid)
        if optimize_widths is not None:
            self.optimize_widths = optimize_widths
        else:
            self.optimize_widths = self._test_gaussianity()
            print("Optimizing widths: ", self.optimize_widths)
        self.shifts = self._find_shifts()
        self.widths = self._find_widths(optimization=self.optimize_widths)
        self.params = self._get_params()

    def _find_shifts(self):
        mu = np.average(self.z, weights=self.nz_mean)
        shifts = [
            (np.average(self.z, weights=nz) - mu) for nz in self.nzs
        ]  # mean of each nz
        return shifts

    def _find_widths(self, optimization=False):
        if optimization:
            return self._find_widths_optimization()
        else:
            return self._find_widths_gaussian()

    def _find_widths_gaussian(self):
        stds = []
        for nz in self.nzs:
            mu = np.average(self.z, weights=nz)
            std = np.sqrt(np.average((self.z - mu) ** 2, weights=nz))
            stds.append(std)
        stds = np.array(stds)
        mu_mean = np.average(self.z, weights=self.nz_mean)
        std_mean = np.sqrt(
            np.average((self.z - mu_mean) ** 2, weights=self.nz_mean)
        )
        widths = stds / std_mean
        return widths

    def _find_widths_optimization(self):
        widths = []
        shifts = self.shifts
        def func(width, nz=np.zeros_like(self.z), shift=0):
            new_nz = shift_and_width_model(self.z, nz, shift, width)
            diff = nz - new_nz
            return np.sum(diff**2)
        for i, nz in enumerate(self.nzs):
            shift = shifts[i]
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(func, args=(nz, shift), bounds=(0.5, 1.5), method='bounded')
            widths.append(res.x)
        return np.array(widths)

    def _test_gaussianity(self):
        # This tests whether the n(z) distribution is close to a Gaussian
        # by measuring the difference between the n(z) and a Gaussian
        # with same mean and stddev. Then we measure the expected value
        # of z under the absolute difference distribution.
        nz_mean = self.nz_mean
        z = self.z
        mu = np.average(self.z, weights=self.nz_mean)
        sigma = np.sqrt(np.average((self.z - mu)**2, weights=self.nz_mean))
        gaussian_nz = np.exp(-0.5 * ((self.z - mu) / sigma) ** 2)
        gaussian_nz /= np.sum(gaussian_nz)
        d = nz_mean - gaussian_nz
        _mu = np.average(z, weights=np.abs(d))
        diff = np.abs(mu - _mu)/mu
        return diff > 0.01

    def _get_prior(self):
        params = self._get_params().T
        mean = np.mean(params, axis=0)
        cov = np.cov(params, rowvar=False)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol
        self.prior_transform = chol

    def _get_params(self):
        return np.array([self.shifts, self.widths])

    def _get_params_names(self):
        return ["delta_z", "width_z"]
