import numpy as np
import scipy.stats as sps
from scipy.special import logsumexp

def mixture_pdf(x, params, weights):
    """
    pdf of a mixture of skew normal distributions
    
    $ f(x) = a * f_1(x) + (1-a) f_0(x) $ 
    """
    return joint_densities(x, params, weights).sum(axis=0)

def joint_densities(x, params, weights):
    """
    weighted pdfs of a mixture of skew normal distributions
    """
    return np.array([w * sps.skewnorm.pdf(x, a, loc, scale) for (a, loc, scale), w in zip(params, weights)])

# def unstable_posteriors(x, params, weights):
#     """
#     posterior probabilities of each component given x
#     """
#     joint_densities_ = joint_densities(x, params, weights)
#     if np.isnan(joint_densities_).any():
#         raise ValueError(f"joint_densities_ contains NaNs: {joint_densities_}")
#     return joint_densities_ / joint_densities_.sum(axis=0, keepdims=True)


def component_posteriors(x, params, individual_sample_weights):
    individual_sample_weights = np.array(individual_sample_weights)[:,None]
    assert len(params) == individual_sample_weights.shape[0]
    log_pdfs = np.stack([sps.skewnorm.logpdf(x.ravel(), *p) for p in params],axis=0)
    numerators = np.zeros_like(log_pdfs)
    numerators = log_pdfs + np.log(individual_sample_weights)
    d = np.zeros_like(numerators[0])
    d = logsumexp(numerators,axis=0)
    P = np.exp(numerators-d[None])
    P[np.isnan(P)] = 0
    return P

def canonical_to_alternate(a, loc, scale):
    """
    convert canonical parameters to alternate parameters

    Arguments:
    a: skewness parameter
    loc: location parameter
    scale: scale parameter

    Returns:
    Delta
    Gamma
    """
    Delta = 0
    Gamma = 0
    
    _delta = a / np.sqrt(1 + a**2)
    Delta = scale * _delta
    Gamma = scale**2 - Delta**2

    return tuple(map(float,(loc,Delta, Gamma)))

def alternate_to_canonical(loc,Delta, Gamma):
    """
    convert alternate parameters to canonical parameters

    Arguments:
    loc: location parameter
    Delta
    Gamma

    Returns:
    a: skewness parameter
    loc: location parameter
    scale: scale parameter
    """
    try:
        a = np.sign(Delta) * np.sqrt(Delta**2 / Gamma)
    except ZeroDivisionError:
        raise ZeroDivisionError(f"Invalid skewness parameter: {np.sign(Delta) * np.sqrt(Delta**2 / Gamma)} from Delta: {Delta}, Gamma: {Gamma}")
    if np.isinf(a) or np.isnan(a):
        raise ZeroDivisionError(f"Invalid skewness parameter: {a} from Delta: {Delta}, Gamma: {Gamma}")
    scale = np.sqrt(Gamma + Delta**2)
    return tuple(map(float,(a, loc, scale)))

def _get_delta(params):
    a = params[0]
    return a / np.sqrt(1 + a**2)