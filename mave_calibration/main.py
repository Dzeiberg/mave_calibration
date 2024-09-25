from mave_calibration.initializations import constrained_gmm_init, gmm_init
from mave_calibration.em_opt.utils import get_sample_weights,constrained_em_iteration,em_iteration,get_likelihood
from mave_calibration.em_opt.constraints import density_constraint_violated

import scipy.stats as sp
import logging
import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm.autonotebook import tqdm
import json
import os
from pathlib import Path
from fire import Fire
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import datetime
from typing import List, Tuple, Iterable

def draw_sample(params : List[Tuple[float]], weights : np.ndarray, sample_size : int=1) -> np.ndarray:
    """
    Draw a list of samples from a mixture of skew normal distributions

    Required Arguments:
    --------------------------------
    params -- List[Tuple[float]] len(NComponents)
        The parameters of the skew normal components
    weights -- Ndarray (NComponents,)
        The mixture weights of the components

    Optional Arguments:
    --------------------------------
    sample_size -- int (default 1)
        The number of observations to draw
    
    Returns:
    --------------------------------
    samples -- Ndarray (sample_size,)
        The drawn sample
    """
    samples = []
    for i in range(sample_size):
        k = np.random.binomial(1,weights[1])
        samples.append(sp.skewnorm.rvs(*params[k]))
    return np.array(samples)

Fit = namedtuple('Fit', ['component_params', 'weights', 'likelihoods'])

def singleFit(observations : np.ndarray, sample_indicators : np.ndarray, **kwargs) -> Fit:
    """
    Run a single fit of the model

    Required Arguments:
    --------------------------------
    observations -- the assay scores (N,)
    sample_indicators -- the sample matrix (N,S) where S[i,j] is True if observation i is in sample j

    Optional Arguments:
    --------------------------------
    constrained -- bool (default True)
        Whether to enforce the density constraint on each sequential pair of components
    buffer_stds -- float (default 1)
        The number of standard deviations to buffer the observation range by when checking the density constraint
    n_components -- int (default 2)
        The number of skew normal components in the model
    max_iters -- int (default 10000)
        The maximum number of EM iterations to perform
    verbose -- bool (default True)
        Whether to display a progress bar
    init_to_sample -- int (default None)
        If not None, initialize the model to the sample index provided, otherwise initialize to all samples
    
    Returns:
    --------------------------------
    component_params -- Tuple[float] len(NSamples)
        the parameters of the skew normal components
    weights -- Ndarray (NSamples, NComponents)
        the mixture weights of each sample
    likelihoods -- Ndarray (NIters,)
        the likelihood of the model at each iteration
    """
    CONSTRAINED = kwargs.get("constrained", True)
    buffer_stds = kwargs.get('buffer_stds',0)
    if buffer_stds < 0:
        raise ValueError("buffer_stds must be non-negative")
    obs_std = observations.std()
    xlims = (observations.min() - obs_std * buffer_stds,
             observations.max() + obs_std * buffer_stds)
    N_components = kwargs.get("n_components", 2)
    assert N_components == 2
    N_samples = sample_indicators.shape[1]
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    # Initialize the components
    if CONSTRAINED:
        init_to_sample = kwargs.get("init_to_sample", None)
        if init_to_sample is not None:
            xInit = observations[sample_indicators[:,init_to_sample]]
        else:
            xInit = observations
        initial_params = constrained_gmm_init(xInit,**kwargs)
        if density_constraint_violated(*initial_params, xlims):
            logging.warning(f"failed to initialized components\nfinal parameters {initial_params[0]}\n{initial_params[1]}\nreturning -inf likelihood")
            return Fit(component_params=initial_params, weights=np.ones((N_samples, N_components)) / N_components, likelihoods=[-1 * np.inf,])
    else:
        initial_params = gmm_init(observations,**kwargs)
    # Initialize the mixture weights of each sample
    W = np.ones((N_samples, N_components)) / N_components
    W = get_sample_weights(observations, sample_indicators, initial_params, W)
    # initial likelihood
    likelihoods = np.array(
        [
            get_likelihood(observations, sample_indicators, initial_params, W) / len(sample_indicators),
        ]
    )
    # Check for bad initialization
    try:
        if CONSTRAINED:
            updated_component_params, updated_weights = (
                constrained_em_iteration(observations, sample_indicators, initial_params, W, xlims, iterNum=0)
            )
        else:
            updated_component_params, updated_weights = em_iteration(
                observations, sample_indicators, initial_params, W
            )
    except ZeroDivisionError:
        logging.warning("ZeroDivisionError")
        return Fit(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf])
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(observations, sample_indicators, updated_component_params, updated_weights)
            / len(sample_indicators),
        ]
    )
    # Run the EM algorithm
    if kwargs.get("verbose",True):
        pbar = tqdm(total=MAX_N_ITERS)
    for i in range(MAX_N_ITERS):
        try:
            if CONSTRAINED:
                updated_component_params, updated_weights = (
                    constrained_em_iteration(
                        observations,sample_indicators, updated_component_params, updated_weights, xlims, iterNum=i+1,
                    )
                )
            else:
                updated_component_params, updated_weights = em_iteration(
                    observations,sample_indicators, updated_component_params, updated_weights
                )
        except ZeroDivisionError:
            print("ZeroDivisionError")
            return Fit(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf])
        likelihoods = np.array(
            [
                *likelihoods,
                get_likelihood(
                    observations,sample_indicators, updated_component_params, updated_weights
                )
                / len(sample_indicators),
            ]
        )
        if kwargs.get("verbose",True):
            pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
            pbar.update(1)
        if i > 51 and (np.abs(likelihoods[-50:] - likelihoods[-51:-1]) < 1e-10).all():
            break
    if kwargs.get("verbose",True):
        pbar.close()
    if CONSTRAINED:
        assert not density_constraint_violated(
            updated_component_params[0], updated_component_params[1], xlims
        )
    return Fit(component_params=updated_component_params, weights=updated_weights, likelihoods=likelihoods)


def prior_from_weights(weights : np.ndarray, population_idx : int=2, controls_idx : int=1, pathogenic_idx : int=0) -> float:
    """
    Calculate the prior probability of an observation from the population being pathogenic

    Required Arguments:
    --------------------------------
    weights -- Ndarray (NSamples, NComponents)
        The mixture weights of each sample

    Optional Arguments:
    --------------------------------
    population_idx -- int (default 2)
        The index of the population component in the weights matrix
    
    controls_idx -- int (default 1)
        The index of the controls (i.e. benign) component in the weights matrix

    pathogenic_idx -- int (default 0)
        The index of the pathogenic component in the weights matrix

    Returns:
    --------------------------------
    prior -- float
        The prior probability of an observation from the population being pathogenic
    """
    prior = ((weights[population_idx, 0] - weights[controls_idx, 0]) / (weights[pathogenic_idx, 0] - weights[controls_idx, 0])).item()
    return np.clip(prior, 1e-10, 1 - 1e-10)

def save(sample_names : List[str], best_fit : Fit, bootstrap_indices : Iterable[Iterable[int]],**kwargs) -> None:
    """
    Save the fit

    Required Arguments:
    --------------------------------
    sample_names -- List[str]
        The sample names

    best_fit -- Fit (i.e. Tuple[Tuple[float], Ndarray, Ndarray])
        The component parameters, weights, and likelihoods of the best fit
    
    bootstrap_indices -- Iterable[Iterable[int]]
        The indices of the bootstrap samples

    Required Keyword Arguments:
    --------------------------------
    save_path -- str
        The path to save the results to

    Optional Keyword Arguments:
    --------------------------------
    dataset_id -- str
        Identifier of the dataset
    """
    save_path = Path(kwargs.get('save_path'))
    save_path.mkdir(exist_ok=True, parents=True)
    filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    dataset_id = kwargs.get("dataset_id",None)
    if dataset_id is not None:
        filename += "_" + dataset_id
    filename += ".json"
    # component_params, weights, likelihoods = best_fit
    savevals = {
                "component_params": best_fit.component_params,
                "weights": best_fit.weights.tolist(),
                "likelihoods": best_fit.likelihoods.tolist(),
                "config": {k:v for k,v in kwargs.items() if is_serializable(v)},
                "sample_names": sample_names,
                "bootstrap_indices": [indices.tolist() for indices in bootstrap_indices],
            }
    for k in savevals:
        if not is_serializable(savevals[k]):
            raise ValueError(f"Value {k} is not serializable")
    with open(os.path.join(save_path, filename), "w") as f:
        json.dump(
            savevals,
            f,
        )

def is_serializable(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False
    
def bootstrap(X : np.ndarray, S : np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int]]]:
    """
    Bootstrap the data

    Required Arguments:
    --------------------------------
    X -- the assay scores (N,)
    S -- the sample matrix (N,S)

    Returns:
    --------------------------------
    XBootstrap -- the bootstrapped assay scores (N,)
    SBootstrap -- the bootstrapped sample matrix (N,S)
    bootstrap_indices -- List[Tuple[int]] -- The indices of the bootstrapped data
    """
    XBootstrap, SBootstrap = X.copy(), S.copy()
    bootstrap_indices = []
    offset = 0
    for sample in range(S.shape[1]):
        sample_bootstrap_indices = np.random.choice(np.where(S[:,sample])[0], size=S[:,sample].sum(), replace=True)
        bootstrap_indices.append(np.array(list(map(lambda a: a.item(), sample_bootstrap_indices))))
        XBootstrap[offset:offset + len(sample_bootstrap_indices)] = X[sample_bootstrap_indices]
        SBootstrap[offset:offset + len(sample_bootstrap_indices), sample] = True
        offset += len(sample_bootstrap_indices)
    return XBootstrap, SBootstrap, bootstrap_indices

def load_data(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read a csv file containing the assay scores and sample names for all observations, e.g.:

    ---------------------------
    Required Keyword Arguments:
    data_filepath : str
        The path to the data file (csv) containing columns sample_name, score for each observation, assumed to contain header

    Returns:
    X -- the assay scores (N,)
    S -- the sample matrix (N,S)
    sample_names -- the sample names (S,)
    """
    data = pd.read_csv(kwargs.get("data_filepath"))
    assert 'score' in data.columns and 'sample_name' in data.columns, f"data file must contain columns 'score', 'sample_name', not {data.columns}"
    sample_names = data.sample_name.unique().tolist()
    X = data.score.values
    S = np.zeros((X.shape[0], len(sample_names)), dtype=bool)
    for i, sample_name in enumerate(sample_names):
        S[:, i] = data.sample_name == sample_name
    assert (S.sum(0) > 0).all(), "each sample must have at least one observation"
    return X, S, sample_names

def run(**kwargs) -> Fit:
    """
    Fit the multi-sample skew normal mixture model to the data
    
    Required Keyword Arguments
    --------------------------
    data_filepath : str
        The path to the data file (csv) containing columns sample_name, score for each observation

    Optional Keyword Arguments:
    --------------------------
    save_path : str
        The path to save the results to
    
    num_fits : int (default 25)
        The number of model fits to perform, choosing the best fit based on likelihood
    
    bootstrap : bool (default True)
        Whether to bootstrap the data before fitting the model

    core_limit : int (default -1)
        The number of cores to use for parallelizing the model fits (n=num_fits), -1 uses all available cores

    Returns:
    --------------------------
    best_fit -- Fit
        The component parameters, weights, and likelihoods of the best fit
    """
    observations, sample_indicators, sample_names = load_data(**kwargs)
    bootstrap_indices = [np.where(Si)[0] for Si in sample_indicators.T]
    if kwargs.get("bootstrap",True):
        observations,sample_indicators,bootstrap_indices = bootstrap(observations,sample_indicators,**kwargs)
    NUM_FITS = kwargs.get("num_fits", 25)
    save_path = kwargs.get("save_path", None)
    best_fit = None
    best_likelihood = -np.inf
    fit_results = Parallel(n_jobs=kwargs.get('core_limit',-1))(delayed(singleFit)(observations, sample_indicators, **kwargs) for i in range(NUM_FITS))
    for fit in fit_results:
        if fit.likelihoods[-1] > best_likelihood:
            best_fit = fit
            best_likelihood = fit.likelihoods[-1]
    if np.isinf(best_likelihood):
        raise ValueError("Failed to fit model")
    if save_path is not None:
        save(sample_names, best_fit, bootstrap_indices,**kwargs)
    return best_fit

if __name__ == "__main__":
    Fire(run)
