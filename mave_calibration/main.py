from mave_calibration.initializations import constrained_gmm_init, gmm_init, random_init, kmeans_init
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
import fire
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import datetime
from typing import List, Tuple, Iterable
import random
from ast import literal_eval

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

def weightConstraints(observations, sample_indicators, W0,benignSampleIdx,constraints,**kwargs):
    functionally_normal_index = W0[benignSampleIdx].argmax() # benign variants are mostly functionally normal
    for sampleNumber, fixedFractionNormal in constraints:
        W0[sampleNumber, functionally_normal_index] = fixedFractionNormal
        W0[sampleNumber, 1-functionally_normal_index] = 1 - fixedFractionNormal
    return W0

def single_fit(observations,sample_indicators, sample_order,**kwargs):
    CONSTRAINED=kwargs.get("Constrained",True)
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    verbose = kwargs.get("verbose",True)
    xlims= (observations.min(),observations.max())
    N_samples = sample_indicators.shape[1]
    N_components = 2
    W = np.ones((N_samples, N_components)) / N_components
    try:
        initial_params,kmeans = kmeans_init(observations, n_clusters=N_components)
    except ValueError:
        logging.warning("Failed to initialize")
        return dict(component_params=[[] for _ in range(N_components)],
                    weights=W,
                    likelihoods=[-1 * np.inf])
    W = get_sample_weights(observations, sample_indicators, initial_params, W)
    constraints = []
    try:
        idxSynonymous = sample_order.index("synonymous")
    except ValueError:
        idxSynonymous = None
    if idxSynonymous is not None:
        constraints.append((idxSynonymous, 1.0),)
    W = weightConstraints(observations, sample_indicators, W, sample_order.index('B/LB'),constraints,**kwargs)
    history = [dict(component_params=initial_params, weights=W)]
    # initial likelihood
    likelihoods = np.array(
        [
            get_likelihood(observations, sample_indicators, initial_params, W) / len(sample_indicators),
        ]
    )
    # Check for bad initialization
    try:
        updated_component_params, updated_weights = (
            constrained_em_iteration(observations, sample_indicators, initial_params, W, xlims, iterNum=0)
        )
    except ZeroDivisionError:
        logging.warning("ZeroDivisionError")
        return dict(component_params=initial_params, weights=W, likelihoods=[*likelihoods, -1 * np.inf],kmeans=kmeans)
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(observations, sample_indicators, updated_component_params, updated_weights)
            / len(sample_indicators),
        ]
    )
    # Run the EM algorithm
    if verbose:
        pbar = tqdm(total=MAX_N_ITERS,leave=False,desc="EM Iteration")
    
    for i in range(MAX_N_ITERS):
        history.append(dict(component_params=updated_component_params, weights=updated_weights))
        if np.isnan(likelihoods).any():
            raise ValueError()
        if np.isnan(np.concatenate(updated_component_params)).any():
            raise ValueError()
        if np.isnan(updated_weights).any():
            raise ValueError()
        if np.isnan(np.concatenate(updated_component_params)).any():
            raise ValueError(f"NaN in updated component params at iteration {i}\n{updated_component_params}")
        if np.isnan(updated_weights).any():
            raise ValueError(f"NaN in updated weights at iteration {i}\n{updated_weights}")
        # observations = np.array([np.random.choice(observation_replicates) for observation_replicates in replicates]).reshape(-1,)
        # try:
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
        if kwargs.get('early_stopping',True) and i >= 1 and (np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-10).all():
            break
    history.append(dict(component_params=updated_component_params, weights=updated_weights))
    if kwargs.get("verbose",True):
        pbar.close()
    if CONSTRAINED:
        assert not density_constraint_violated(
            updated_component_params[0], updated_component_params[1], xlims
        )
        
    return dict(component_params=updated_component_params, weights=updated_weights, likelihoods=likelihoods, history=history,kmeans=kmeans)

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
                "component_params": best_fit.get('component_params'),
                "weights": best_fit.get('weights').tolist(),
                "likelihoods": best_fit.get('likelihoods').tolist(),
                "config": {k:v for k,v in kwargs.items() if is_serializable(v)},
                "observations": observations,
                "sample_indicators": sample_indicators.tolist(),
                "sample_order": sample_order,
                "bootstrap_indices": bootstrap_indices.tolist(),
                "history": [(hist['component_params'], hist['weights'].tolist()) for hist in best_fit.get("history",[])],
                "kmeans_centers": best_fit['kmeans'].cluster_centers_.tolist() if 'kmeans' in best_fit else None
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

def downsample(observations, sample_indicators, proportion):
    NSamples = sample_indicators.sum(0)
    downsampleNum = np.round(NSamples * proportion).astype(int)
    indices = np.concatenate([np.random.choice(np.where(sample_indicators[:,i])[0], size=downsampleNum[i], replace=False) for i in range(sample_indicators.shape[1])])
    downsampled_observations = observations[indices]
    downsampled_sample_indicators = sample_indicators[indices]
    return downsampled_observations, downsampled_sample_indicators, indices

def run(data_filepath, **kwargs) -> Fit:
    """
    Fit the multi-sample skew normal mixture model to the data
    
    Required Arguments
    --------------------------
    data_filepath : str
        The path to the data file (csv) containing columns sample_name, score for each observation
    
    Optional Keyword Arguments:
    --------------------------

    save_path : str
        The path to save the results to
    
    num_fits : int (default 25)
        The number of model fits to perform, choosing the best fit based on likelihood

    core_limit : int (default -1)
        The number of cores to use for parallelizing the model fits (n=num_fits), -1 uses all available cores
    
    downsample_proportion : float (default 1.0)
        The proportion of the data to use for fitting the model

    Returns:
    --------------------------
    best_fit -- Fit
        The component parameters, weights, and likelihoods of the best fit
    """
    observations, sample_indicators, sample_order, bootstrap_indices = prep_data(data_filepath,**kwargs)
    downsample_proportion = kwargs.get("downsample_proportion",1.0)
    if downsample_proportion < 1.0:
        observations, sample_indicators, bootstrap_indices = downsample(observations, sample_indicators, downsample_proportion)
    NUM_FITS = kwargs.get("num_fits", 25)
    save_path = kwargs.get("save_path", None)
    best_fit = None
    best_likelihood = -np.inf
    cores = kwargs.get('core_limit',-1)
    if cores == 1:
        fit_results = [runFitIteration(observations, sample_indicators, sample_order,**kwargs) for i in range(NUM_FITS)]
    else:
        fit_results = Parallel(n_jobs=kwargs.get('core_limit',-1))(delayed(runFitIteration)(observations,
                                                                                            sample_indicators, sample_order,**kwargs) \
                                                                        for i in range(NUM_FITS))
    for (fit,fit_likelihood) in fit_results:
        if fit_likelihood > best_likelihood:
            best_fit = fit
            best_likelihood = fit_likelihood
    if np.isinf(best_likelihood):
        raise ValueError("Failed to fit model")
    if save_path is not None:
        save(observations,sample_indicators,sample_order, bootstrap_indices,best_fit,**kwargs)
    # return best_fit

def runFitIteration(observations, sample_indicators, sample_order, **kwargs):
    bootstrap_sample_indices = [np.random.choice(np.where(sample_i)[0],
                                                         sample_i.sum(),
                                                         replace=True) for sample_i in sample_indicators.T]
    indices = np.concatenate(bootstrap_sample_indices)
    try:
        iter_fit = single_fit(observations[indices],sample_indicators[indices],sample_order,**kwargs)
    except AssertionError:
        iter_fit = dict(component_params=None, weights=None, likelihoods=[-1 * np.inf])
        return iter_fit, -1 * np.inf
    iteration_ll = get_likelihood(observations,sample_indicators,
                                    iter_fit.get('component_params'),
                                    iter_fit.get('weights'))/len(observations)
    return iter_fit, iteration_ll

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
                "auth_reported_score": float,
                "labels": List[str]
            },
            {
                "auth_reported_score": float,
                "labels": List[str]
            },
            ...
        ]
    """
    restarts = 0
    all_samples_represented = False
    data = pd.read_csv(data_filepath).assign(labels=lambda x: x.labels.apply(literal_eval))
    missing_data = data.auth_reported_score.isna() | data.labels.isna()
    if missing_data.sum() > 0:
        logging.warning(f"Missing data in {missing_data.sum()} observations")
        data = data[~missing_data]
    while not all_samples_represented and restarts < 100:
        assert 'auth_reported_score' in data.columns and 'labels' in data.columns, f"data file must contain columns 'auth_reported_score', 'labels', not {data.columns}"
        # names of the labels that are options to model
        label_options = {"P/LP",'B/LB','gnomAD','synonymous'}
        # get records that are candidates for inclusion
        candidate_observations = data[data.labels.apply(lambda x: len(set(x).intersection(label_options)) > 0)]
        # if the variant is synonymous, assign it that label, otherwise randomly choose one of P/LP, B/LB, or gnomAD
        includes_synonymous = candidate_observations['labels'].apply(lambda x: 'synonymous' in x).sum() > 0
        if not includes_synonymous:
            label_options = label_options - {"synonymous"}
        repeat = 0
        chosen_label = []
        while len(set(chosen_label)) < len(label_options) and repeat < 100:
            chosen_label = []
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
            repeat += 1
        assert len(set(chosen_label)) == len(label_options), f"failed to assign all labels after {repeat} attempts"
        # assign the randomly chosen label
        candidate_observations = candidate_observations.assign(chosen_label=chosen_label)

        # choose variants to include in bootstrap sample
        bootstrap_indices = np.random.randint(0, len(candidate_observations), size=(len(candidate_observations),))
        bootstraped_data = candidate_observations.iloc[bootstrap_indices]
        # observations = np.array(list(bootstraped_data.scores.apply(lambda replicates: [np.mean(replicates),] if not use_replicates else replicates).values)).ravel()
        observations = np.array(list(bootstraped_data.auth_reported_score.values))
        labels = bootstraped_data.chosen_label.values.tolist()
        # create the sample indicator matrix
        NSamples = len(set(label_options))
        sample_indicators = np.zeros((len(observations), NSamples), dtype=bool)
        sample_names = ["P/LP",'B/LB','gnomAD']
        if includes_synonymous:
            sample_names.append("synonymous")
        labels = np.array(labels)
        for i, label_val in enumerate(sample_names):
            sample_indicators[:, i] = labels == label_val
        # remove any samples that are all zeros
        all_samples_represented = (sample_indicators.sum(0) > 0).all()
    if all_samples_represented:
        assert (sample_indicators.sum(0) > 0).all(), "each sample must have at least one observation; current counts: " + str(sample_indicators.sum(0))
        assert (sample_indicators.sum(1) == 1).all(), "each observation must belong to exactly one sample"
        assert sample_indicators.shape[0] == len(observations), "number of observations must match number of sample indicators"
        return observations, sample_indicators, sample_names, bootstrap_indices
    else:
        raise ValueError("Failed to represent all samples")


if __name__ == '__main__':
#   fire.Fire(run)
    run(data_filepath="/data/dzeiberg/IGVF-cvfg-pillar-project/Pillar_project_data_files/individual_datasets/MSH2_Jia_2021.csv",
          num_fits=10,core_limit=10,save_path="/data/dzeiberg/mave_calibration/test_results_10_22_24/")
