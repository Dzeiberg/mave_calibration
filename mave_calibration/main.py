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
import random

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

# def singleFit(observations : np.ndarray, sample_indicators : np.ndarray, **kwargs) -> Fit:
#     """
#     Run a single fit of the model

#     Required Arguments:
#     --------------------------------
#     observations -- the assay scores (N_Observations)
#     sample_indicators -- the sample matrix (N_ObservationsS) where S[i,j] is True if observation i is in sample j

#     Optional Arguments:
#     --------------------------------
#     constrained -- bool (default True)
#         Whether to enforce the density constraint on each sequential pair of components
#     n_components -- int (default 2)
#         The number of skew normal components in the model
#     max_iters -- int (default 10000)
#         The maximum number of EM iterations to perform
#     verbose -- bool (default True)
#         Whether to display a progress bar
#     init_to_sample -- int (default None)
#         If not None, initialize the model to the sample index provided, otherwise initialize to all samples
#     replicates -- List[List[float]] (default None)
#         If not None, the replicates of the assay scores
#     Returns:
#     --------------------------------
#     component_params -- Tuple[float] len(NSamples)
#         the parameters of the skew normal components
#     weights -- Ndarray (NSamples, NComponents)
#         the mixture weights of each sample
#     likelihoods -- Ndarray (NIters,)
#         the likelihood of the model at each iteration
#     """
#     CONSTRAINED = kwargs.get("constrained", True)
#     buffer_stds = kwargs.get('buffer_stds',0)
#     if buffer_stds < 0:
#         raise ValueError("buffer_stds must be non-negative")
#     xlims = (observations.min(),observations.max())
#     N_components = kwargs.get("n_components", 2)
#     assert N_components == 2
#     N_samples = sample_indicators.shape[1]
#     MAX_N_ITERS = kwargs.get("max_iters", 10000)
#     # Initialize the components
#     if CONSTRAINED:
#         initial_params = constrained_gmm_init(observations,sample_indicators,**kwargs)
#         if density_constraint_violated(*initial_params, xlims):
#             logging.warning(f"failed to initialized components\nfinal parameters {initial_params[0]}\n{initial_params[1]}\nreturning -inf likelihood")
#             return Fit(component_params=initial_params, weights=np.ones((N_samples, N_components)) / N_components, likelihoods=[-1 * np.inf,])
#     else:
#         initial_params = gmm_init(observations,**kwargs)
#     # Initialize the mixture weights of each sample
#     W = np.ones((N_samples, N_components)) / N_components
#     W = get_sample_weights(observations, sample_indicators, initial_params, W)
#     # initial likelihood
#     likelihoods = np.array(
#         [
#             get_likelihood(observations, sample_indicators, initial_params, W) / len(sample_indicators),
#         ]
#     )
#     # Check for bad initialization
#     try:
#         if CONSTRAINED:
#             updated_component_params, updated_weights = (
#                 constrained_em_iteration(observations, sample_indicators, initial_params, W, xlims, iterNum=0)
#             )
#         else:
#             updated_component_params, updated_weights = em_iteration(
#                 observations, sample_indicators, initial_params, W
#             )
#     except ZeroDivisionError:
#         logging.warning("ZeroDivisionError")
#         return Fit(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf])
#     likelihoods = np.array(
#         [
#             *likelihoods,
#             get_likelihood(observations, sample_indicators, updated_component_params, updated_weights)
#             / len(sample_indicators),
#         ]
#     )
#     # Run the EM algorithm
#     if kwargs.get("verbose",True):
#         pbar = tqdm(total=MAX_N_ITERS)
#     for i in range(MAX_N_ITERS):
#         try:
#             if CONSTRAINED:
#                 updated_component_params, updated_weights = (
#                     constrained_em_iteration(
#                         observations,sample_indicators, updated_component_params, updated_weights, xlims, iterNum=i+1,
#                     )
#                 )
#             else:
#                 updated_component_params, updated_weights = em_iteration(
#                     observations,sample_indicators, updated_component_params, updated_weights
#                 )
#         except ZeroDivisionError:
#             print("ZeroDivisionError")
#             return Fit(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf])
#         likelihoods = np.array(
#             [
#                 *likelihoods,
#                 get_likelihood(
#                     observations,sample_indicators, updated_component_params, updated_weights
#                 )
#                 / len(sample_indicators),
#             ]
#         )
#         if kwargs.get("verbose",True):
#             pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
#             pbar.update(1)
#         if kwargs.get('early_stopping',True) and i > 51 and (np.abs(likelihoods[-50:] - likelihoods[-51:-1]) < 1e-10).all():
#             break
#     if kwargs.get("verbose",True):
#         pbar.close()
#     if CONSTRAINED:
#         assert not density_constraint_violated(
#             updated_component_params[0], updated_component_params[1], xlims
#         )
#     return Fit(component_params=updated_component_params, weights=updated_weights, likelihoods=likelihoods)

def singleFit(replicates : List[List[float]], sample_indicators : np.ndarray, **kwargs) -> Fit:
    """
    Run a single fit of the model

    Required Arguments:
    --------------------------------
    replicates -- List[List[float]] -- the assay scores (N_Observations)

    sample_indicators -- the sample matrix (N_ObservationsS) where S[i,j] is True if observation i is in sample j

    Optional Arguments:
    --------------------------------
    constrained -- bool (default True)
        Whether to enforce the density constraint on each sequential pair of components
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
    all_observations = np.concatenate(replicates).reshape(-1)
    avg_score = np.array([np.mean(replicate) for replicate in replicates])
    xlims = (all_observations.min(),all_observations.max())
    N_components = kwargs.get("n_components", 2)
    assert N_components == 2
    N_samples = sample_indicators.shape[1]
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    # Initialize the components
    if CONSTRAINED:
        initial_params = constrained_gmm_init(replicates,sample_indicators,**kwargs)
        if density_constraint_violated(*initial_params, xlims):
            logging.warning(f"failed to initialized components\nfinal parameters {initial_params[0]}\n{initial_params[1]}\nreturning -inf likelihood")
            return Fit(component_params=initial_params, weights=np.ones((N_samples, N_components)) / N_components, likelihoods=[-1 * np.inf,])
    else:
        initial_params = gmm_init(all_observations,**kwargs)
    # Initialize the mixture weights of each sample
    W = np.ones((N_samples, N_components)) / N_components
    W = get_sample_weights(avg_score, sample_indicators, initial_params, W)
    # initial likelihood
    likelihoods = np.array(
        [
            get_likelihood(avg_score, sample_indicators, initial_params, W) / len(sample_indicators),
        ]
    )
    # Check for bad initialization
    try:
        if CONSTRAINED:
            updated_component_params, updated_weights = (
                constrained_em_iteration(avg_score, sample_indicators, initial_params, W, xlims, iterNum=0)
            )
        else:
            updated_component_params, updated_weights = em_iteration(
                avg_score, sample_indicators, initial_params, W
            )
    except ZeroDivisionError:
        logging.warning("ZeroDivisionError")
        return Fit(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf])
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(avg_score, sample_indicators, updated_component_params, updated_weights)
            / len(sample_indicators),
        ]
    )
    # Run the EM algorithm
    if kwargs.get("verbose",True):
        pbar = tqdm(total=MAX_N_ITERS)
    indicators = np.concatenate([np.tile(sample_indicators[j], (len(replicates[j]),1)) for j in range(len(replicates))])
    assert len(indicators) == len(all_observations)
    for i in range(MAX_N_ITERS):
        if np.isnan(np.concatenate(updated_component_params)).any():
            raise ValueError(f"NaN in updated component params at iteration {i}\n{updated_component_params}")
        if np.isnan(updated_weights).any():
            raise ValueError(f"NaN in updated weights at iteration {i}\n{updated_weights}")
        # observations = np.array([np.random.choice(observation_replicates) for observation_replicates in replicates]).reshape(-1,)
        try:
            if CONSTRAINED:
                updated_component_params, updated_weights = (
                    constrained_em_iteration(
                        all_observations,indicators, updated_component_params, updated_weights, xlims, iterNum=i+1,
                    )
                )
            else:
                updated_component_params, updated_weights = em_iteration(
                    all_observations,indicators, updated_component_params, updated_weights
                )
        except ZeroDivisionError:
            print("ZeroDivisionError")
            return Fit(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf])
        likelihoods = np.array(
            [
                *likelihoods,
                get_likelihood(
                    all_observations,indicators, updated_component_params, updated_weights
                )
                / len(indicators),
            ]
        )
        if kwargs.get("verbose",True):
            pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
            pbar.update(1)
        if kwargs.get('early_stopping',True) and i > 51 and (np.abs(likelihoods[-50:] - likelihoods[-51:-1]) < 1e-10).all():
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

def save(observations,sample_indicators,sample_order, bootstrap_indices,best_fit,**kwargs) -> None:
    """
    Save the fit

    Required Arguments:
    --------------------------------
    observations -- Ndarray (N,)
        The assay scores
    sample_indicators -- Ndarray (N,S)
        The sample indicator matrix
    
    sample_order -- List[str] (S,)
        The sample names corresponding to the sample indicator matrix

    bootstrap_indices -- Iterable[int] (N,)
        The indices of the bootstrap samples
        
    best_fit -- Fit (i.e. Tuple[Tuple[float], Ndarray, Ndarray])
        The component parameters, weights, and likelihoods of the best fit
    
    

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
    try:
        observations = observations.tolist()
    except AttributeError:
        pass
    savevals = {
                "component_params": best_fit.component_params,
                "weights": best_fit.weights.tolist(),
                "likelihoods": best_fit.likelihoods.tolist(),
                "config": {k:v for k,v in kwargs.items() if is_serializable(v)},
                "observations": observations,
                "sample_indicators": sample_indicators.tolist(),
                "sample_order": sample_order,
                "bootstrap_indices": bootstrap_indices.tolist(),
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

def do_single(*args,**kwargs):
    try:
        return singleFit(*args,**kwargs)
    except AssertionError:
        return Fit(component_params=None, weights=None, likelihoods=[-1 * np.inf])

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

def run(data_filepath, **kwargs) -> Fit:
    """
    Fit the multi-sample skew normal mixture model to the data
    
    Required Arguments
    --------------------------
    data_filepath : str
        The path to the data file (csv) containing columns sample_name, score for each observation
    
    Optional Keyword Arguments:
    --------------------------
    use_replicates : bool (default False)
        if true, all available replicates will be used for each observation, otherwise the average score will be used

    save_path : str
        The path to save the results to
    
    num_fits : int (default 25)
        The number of model fits to perform, choosing the best fit based on likelihood

    core_limit : int (default -1)
        The number of cores to use for parallelizing the model fits (n=num_fits), -1 uses all available cores

    Returns:
    --------------------------
    best_fit -- Fit
        The component parameters, weights, and likelihoods of the best fit
    """
    observations, sample_indicators, sample_order, bootstrap_indices = prep_data(data_filepath,**kwargs)
    NUM_FITS = kwargs.get("num_fits", 25)
    save_path = kwargs.get("save_path", None)
    best_fit = None
    best_likelihood = -np.inf
    fit_results = Parallel(n_jobs=kwargs.get('core_limit',-1))(delayed(do_single)(observations, sample_indicators, **kwargs) for i in range(NUM_FITS))
    for fit in fit_results:
        if fit.likelihoods[-1] > best_likelihood:
            best_fit = fit
            best_likelihood = fit.likelihoods[-1]
    if np.isinf(best_likelihood):
        raise ValueError("Failed to fit model")
    if save_path is not None:
        save(observations,sample_indicators,sample_order, bootstrap_indices,best_fit,**kwargs)
    return best_fit

def prep_data(data_filepath : str,**kwargs):
    """
    Prepare the data for fitting the model

    Required Arguments:
    --------------------------------
    data_filepath -- str
        The path to the data file (json)
        expected format:
        [
            {
                "scores": List[float],
                "labels": List[str]
            },
            {
                "scores": List[float],
                "labels": List[str]
            },
            ...
        ]

    Optional Arguments:
    --------------------------------
    - use_replicates : bool (default False)
        if true, all available replicates will be used for each observation, otherwise the average score will be used
    """
    use_replicates = kwargs.get("use_replicates", False)
    data = pd.read_json(data_filepath)
    assert 'scores' in data.columns and 'labels' in data.columns, f"data file must contain columns 'scores', 'labels', not {data.columns}"
    # names of the labels that are options to model
    label_options = {"P/LP",'B/LB','gnomAD','synonymous'}
    # get records that are candidates for inclusion
    candidate_observations = data[data.labels.apply(lambda x: len(set(x).intersection(label_options)) > 0)]
    # for each instance, randomly choose one of the replicates (if there are multiple) and assign a label
    # if the variant is synonymous, assign it that label, otherwise randomly choose one of P/LP, B/LB, or gnomAD
    chosen_label = []
    # chosen_replicate = []
    for _,candidate in candidate_observations.iterrows():
        labels = set(candidate.labels).intersection(label_options)
        assert len(labels) > 0
        if len(labels) == 1:
            label = next(iter(labels))
        else:
            if "synonymous" in candidate.labels:
                label = "synonymous"
            else:
                labels = list(labels)
                random.shuffle(labels)
                label = labels[0]
        assert label in label_options, f"label {label} not in {label_options}"
        chosen_label.append(label)
        # chosen_replicate.append(replicates[0])
    # assign the randomly chosen label and replicate to each candidate observation
    candidate_observations = candidate_observations.assign(chosen_label=chosen_label)
    # choose variants to include in bootstrap sample
    bootstrap_indices = np.random.randint(0, len(candidate_observations), size=(len(candidate_observations),))
    bootstraped_data = candidate_observations.iloc[bootstrap_indices]
    # observations = bootstraped_data.chosen_replicate.values
    observations = list(bootstraped_data.scores.apply(lambda replicates: [np.mean(replicates),] if not use_replicates else replicates).values)
    labels = bootstraped_data.chosen_label.values.tolist()
    # create the sample indicator matrix
    unique_labels = ['P/LP','B/LB','gnomAD','synonymous']
    NSamples = len(set(unique_labels))
    sample_indicators = np.zeros((len(observations), NSamples), dtype=bool)
    labels = np.array(labels)
    for i, label_val in enumerate(unique_labels):
        sample_indicators[:, i] = labels == label_val
    # remove any samples that are all zeros
    sample_included = sample_indicators.sum(0) > 0
    sample_indicators = sample_indicators[:,sample_included]
    unique_labels = np.array(unique_labels)[sample_included]
    assert (sample_indicators.sum(0) > 0).all(), "each sample must have at least one observation; current counts: " + str(sample_indicators.sum(0))
    assert (sample_indicators.sum(1) == 1).all(), "each observation must belong to exactly one sample"
    assert sample_indicators.shape[0] == len(observations), "number of observations must match number of sample indicators"
    
    return observations, sample_indicators, unique_labels.tolist(), bootstrap_indices


if __name__ == "__main__":
    Fire(run)
