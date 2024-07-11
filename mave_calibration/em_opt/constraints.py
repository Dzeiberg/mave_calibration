import scipy.stats as sps
import numpy as np
from mave_calibration.skew_normal import density_utils

def density_constraint_violated(params_1, params_2, xlims):
    """
    Check if the density ratio of distribution 1 to distribution 2 is monotonic

    Arguments:
    params_1: tuple : (a, loc, scale) : skewness, location, scale parameters of distribution 1
    params_2: tuple : (a, loc, scale) : skewness, location, scale parameters of distribution 2
    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    bool : True if the density ratio is not monotonic (constraint violated), False otherwise
    """

    log_pdf_1 = sps.skewnorm.logpdf(np.linspace(*xlims, 1000), *params_1)
    log_pdf_2 = sps.skewnorm.logpdf(np.linspace(*xlims, 1000), *params_2)

    return not np.all(np.diff(log_pdf_1 - log_pdf_2) < .01)


def weighted_density_constraint_violated(params, weights, xlims):
    def prior_from_weights(W):
        return ((W[2,0] - W[1,0]) / (W[0,0] - W[1,0])).item()
    rng = np.linspace(*xlims, 1000)
    f_P = density_utils.joint_densities(rng, params, weights[0]).sum(0)
    f_B = density_utils.joint_densities(rng, params, weights[1]).sum(0)
    # a = prior_from_weights(weights)
    # P =  a * f_P / (a * f_P + (1 - a) * f_B)
    log_lr = np.log(f_P) - np.log(f_B)
    return not np.all(np.diff(log_lr) < 0.0)