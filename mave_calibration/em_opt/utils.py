import scipy.stats as sps
import numpy as np
from mave_calibration.skew_normal import density_utils

def trunc_norm_moments(mu, sigma):
    """Array trunc norm moments"""
    cdf = sps.norm.cdf(mu / sigma)
    flags = cdf == 0
    pdf = sps.norm.pdf(mu / sigma)
    p = np.zeros_like(pdf)
    p[~flags] = pdf[~flags] / cdf[~flags]
    p[flags] = abs(mu[flags] / sigma)

    m1 = mu + sigma * p
    m2 = mu ** 2 + sigma ** 2 + sigma * mu * p
    return m1, m2

def get_truncated_normal_moments(observations, component_params):
    _delta = density_utils._get_delta(component_params)
    loc,scale = component_params[1:]
    truncated_normal_loc = _delta / scale * (observations - loc)
    truncated_normal_scale = np.sqrt(1 - _delta**2)
    v,w = trunc_norm_moments(truncated_normal_loc, truncated_normal_scale)
    return v,w

def em_iteration(observations, sample_indicators, current_component_params, current_weights):
    """
    Perform one iteration of the EM algorithm

    Arguments:
    observations: np.array (N,) : observed instances
    sample_indicators: np.array (N, S) : indicator matrix of which sample each observation belongs to
    current_component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : skewness, location, scale parameters of each component
    current_weights: np.array (S, K) : weights of each component for each sample

    Returns:
    updated_component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : updated skewness, location, scale parameters of each component
    updated_weights: np.array (S, K) : updated weights of each component for each sample
    """
    N,S = sample_indicators.shape
    K = len(current_component_params)
    assert current_weights.shape == (S, K)
    sample_indicators = validate_indicators(sample_indicators)
    assert sample_indicators.shape == (N, S)
    responsibilities = sample_specific_responsibilities(observations, sample_indicators, current_component_params, current_weights)
    updated_component_params = []
    for i,curr_comp_params in enumerate(current_component_params): # for each component
        updated_loc = get_location_update(observations, responsibilities[i], curr_comp_params)
        updated_Delta = get_Delta_update(updated_loc, observations, responsibilities[i], curr_comp_params)
        updated_Gamma = get_Gamma_update(updated_loc, updated_Delta, observations, responsibilities[i], curr_comp_params)
        updated_component_params.append(density_utils.alternate_to_canonical(updated_loc, updated_Delta, updated_Gamma))
    updated_weights = get_sample_weights(observations, sample_indicators, updated_component_params, current_weights)
    return updated_component_params, updated_weights

def get_sample_weights(observations, sample_indicators, updated_component_params, current_weights):
    updated_weights = np.zeros_like(current_weights)
    for i in range(current_weights.shape[0]): # for each sample
        sample_observations = observations[sample_indicators[:,i]]
        updated_weights[i] = density_utils.component_posteriors(sample_observations, updated_component_params, current_weights[i]).mean(1)
    return updated_weights

def get_likelihood(observations, sample_indicators, component_params, weights):
    Likelihood = 0.
    for sample_num,sample_mask in enumerate(sample_indicators.T):
        X = observations[sample_mask]
        sample_likelihood = density_utils.joint_densities(X, component_params, weights[sample_num]).sum(axis=0)
        Likelihood += np.log(sample_likelihood).sum().item()
    return Likelihood

def sample_specific_responsibilities(observations , sample_indicators , component_params, weights):
    """
    For each observation calculate the posteriors with respect to each component and that observation's sample's component weights

    Arguments:
    observations: np.array (N,) : observed instances
    sample_indicators: np.array (N, S) : indicator matrix of which sample each observation belongs to
    component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : skewness, location, scale parameters of each component
    weights: np.array (S, K) : weights of each component for each sample

    Returns:
    responsibilities: np.array (K, N) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    """
    N_samples = sample_indicators.shape[1]
    N_components = len(component_params)
    N_observations = len(observations)
    assert weights.shape == (N_samples, N_components)

    responsibilities = np.zeros((N_components, N_observations))
    for i,sample_mask in enumerate(sample_indicators.T):
        X = observations[sample_mask]
        responsibilities[:, sample_mask] = density_utils.component_posteriors(X, component_params, weights[i])
    return responsibilities


def validate_indicators(I):
    assert I.ndim == 2
    assert (I.sum(1) == 1).all()
    assert np.isin(I, [0, 1]).all()
    return I.astype(bool)


def get_location_update(observations, responsibilities, component_params):
    """
    Calculate the location update for the given component

    Arguments:
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_loc: float : updated location parameter
    """
    assert observations.shape == responsibilities.shape
    v,w = get_truncated_normal_moments(observations, component_params)
    (_,Delta, Gamma) = density_utils.canonical_to_alternate(*component_params)
    m = observations - v * Delta
    return (m * responsibilities).sum() / responsibilities.sum()


def get_Delta_update(updated_loc, observations, responsibilities, component_params):
    """
    Calculate the Delta update for the given component

    Arguments:
    updated_loc: float : updated location parameter from this iteration
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_Delta: float : updated Delta parameter
    """

    assert observations.shape == responsibilities.shape
    v,w = get_truncated_normal_moments(observations, component_params)
    d = v * (observations - updated_loc)
    return (d * responsibilities).sum() / responsibilities.sum()

def get_Gamma_update(updated_loc, updated_Delta, observations, responsibilities, component_params):
    """
    Calculate the Gamma update for the given component

    Arguments:
    updated_loc: float : updated location parameter from this iteration
    updated_Delta: float : updated Delta parameter from this iteration
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_Gamma: float : updated Gamma parameter
    """
    assert observations.shape == responsibilities.shape
    v,w = get_truncated_normal_moments(observations, component_params)
    g = (observations - updated_loc)**2  - \
        (2 * updated_Delta * v * (observations - updated_loc)) + \
            (updated_Delta**2 * w)
    return (g * responsibilities).sum() / responsibilities.sum()
