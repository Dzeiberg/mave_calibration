from mave_calibration.initializations import constrained_gmm_init, gmm_init
import scipy.stats as sp
import numpy as np
import pandas as pd
from mave_calibration.em_opt.utils import get_sample_weights,constrained_em_iteration,em_iteration,get_likelihood
from mave_calibration.em_opt.constraints import density_constraint_violated
from tqdm.autonotebook import tqdm
import json
import os
from pathlib import Path
from fire import Fire
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import datetime


def draw_sample(params,weights,N=1):
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
    N -- int (default 1)
        The number of observations to draw
    
    Returns:
    --------------------------------
    samples -- Ndarray (N,)
        The drawn sample
    """
    samples = []
    for i in range(N):
        k = np.random.binomial(1,weights[1])
        samples.append(sp.skewnorm.rvs(*params[k]))
    return np.array(samples)

def singleFit(X, S, **kwargs):
    """
    Run a single fit of the model

    Required Arguments:
    --------------------------------
    X -- the assay scores (N,)
    S -- the sample matrix (N,S) where S[i,j] is True if observation i is in sample j

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
    buffer_stds = kwargs.get('buffer_stds',1)
    if buffer_stds < 0:
        raise ValueError("buffer_stds must be non-negative")
    obs_std = X.std()
    xlims = (X.min() - obs_std * buffer_stds,
             X.max() + obs_std * buffer_stds)
    N_components = kwargs.get("n_components", 2)
    N_samples = S.shape[1]
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    # Initialize the components
    if CONSTRAINED:
        init_to_sample = kwargs.get("init_to_sample", None)
        if init_to_sample is not None:
            xInit = X[S[:,init_to_sample]]
        else:
            xInit = X
        initial_params = constrained_gmm_init(xInit,**kwargs)
        assert not density_constraint_violated(*initial_params, xlims)
    else:
        initial_params = gmm_init(X,**kwargs)
    # Initialize the mixture weights of each sample
    W = np.ones((N_samples, N_components)) / N_components
    W = get_sample_weights(X, S, initial_params, W)
    # initial likelihood
    likelihoods = np.array(
        [
            get_likelihood(X, S, initial_params, W) / len(S),
        ]
    )
    # Check for bad initialization
    try:
        if CONSTRAINED:
            updated_component_params, updated_weights = (
                constrained_em_iteration(X, S, initial_params, W, xlims, iterNum=0)
            )
        else:
            updated_component_params, updated_weights = em_iteration(
                X, S, initial_params, W
            )
    except ZeroDivisionError as e:
        print("ZeroDivisionError")
        return initial_params, W, [*likelihoods, -1 * np.inf]
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(X, S, updated_component_params, updated_weights)
            / len(S),
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
                        X, S, updated_component_params, updated_weights, xlims, iterNum=i+1,
                    )
                )
            else:
                updated_component_params, updated_weights = em_iteration(
                    X, S, updated_component_params, updated_weights
                )
        except ZeroDivisionError as e:
            print("ZeroDivisionError")
            return initial_params, W, [*likelihoods, -1 * np.inf]
        likelihoods = np.array(
            [
                *likelihoods,
                get_likelihood(
                    X, S, updated_component_params, updated_weights
                )
                / len(S),
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
    return updated_component_params, updated_weights, likelihoods


def prior_from_weights(W, population_idx=2, controls_idx=1, pathogenic_idx=0):
    prior = ((W[population_idx, 0] - W[controls_idx, 0]) / (W[pathogenic_idx, 0] - W[controls_idx, 0])).item()
    return np.clip(prior, 1e-10, 1 - 1e-10)


def P2LR(p, alpha):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return p / (1 - p) * (1 - alpha) / alpha
    # return np.log(p) - np.log(1 - p) + np.log(1-alpha) - np.log(alpha)

def LR2P(lr, alpha):
    return 1 / (1 + np.exp(-1 * (np.log(lr) + np.log(alpha) - np.log(1 - alpha))))

def get_fit_statistics(X, S, sample_names, weights, component_params):
    statistics = {}
    for sample_idx, sample_name in enumerate(sample_names):
        sample_observations = X[S[:,sample_idx]]
        synthetic_observations = draw_sample(component_params,weights[sample_idx],N=len(sample_observations))
        ks_test = sp.kstest(sample_observations, synthetic_observations)
        ks_stat, ks_p = ks_test.statistic, ks_test.pvalue
        xU = np.concatenate((sample_observations, synthetic_observations))
        yU = np.concatenate((np.zeros_like(sample_observations), np.ones_like(synthetic_observations)))
        u_stat, u_p = sp.mannwhitneyu(sample_observations, synthetic_observations)
        auc = roc_auc_score(yU, xU)
        statistics[sample_name] = dict(ks_stat=ks_stat, ks_p=ks_p, u_stat=u_stat, u_p=u_p, auc=auc)
    return statistics

def save(loaded_sample_names, best_fit, bootstrap_indices,**kwargs):
    """
    Save the fit

    Required Arguments:
    --------------------------------
    loaded_sample_names -- List[str]
        The sample names

    best_fit -- Tuple[Tuple[float], Ndarray, Ndarray]
        The component parameters, weights, and likelihoods of the best fit
    
    bootstrap_indices -- List[Tuple[int]]
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
    component_params, weights, likelihoods = best_fit
    with open(os.path.join(save_path, filename), "w") as f:
        json.dump(
            {
                "component_params": component_params,
                "weights": weights.tolist(),
                "likelihoods": likelihoods.tolist(),
                "config": kwargs,
                "sample_names": loaded_sample_names,
                "bootstrap_indices": bootstrap_indices,
            },
            f,
        )

def bootstrap(X, S, **kwargs):
    XBootstrap, SBootstrap = X.copy(), S.copy()
    bootstrap_indices = []
    offset = 0
    for sample in range(S.shape[1]):
        sample_bootstrap_indices = np.random.choice(np.where(S[:,sample])[0], size=S[:,sample].sum(), replace=True)
        bootstrap_indices.append(tuple(list(map(lambda a: a.item(), sample_bootstrap_indices))))
        XBootstrap[offset:offset + len(sample_bootstrap_indices)] = X[sample_bootstrap_indices]
        SBootstrap[offset:offset + len(sample_bootstrap_indices), sample] = True
        offset += len(sample_bootstrap_indices)
    return XBootstrap, SBootstrap, bootstrap_indices

def load_data(**kwargs):
    """

    Required Keyword Arguments:
    data_filepath : str
        The path to the data file (csv) containing columns sample_name, score for each observation

    Returns:
    X -- the assay scores (N,)
    S -- the sample matrix (N,S)
    sample_names -- the sample names (S,)
    """
    data = pd.read_csv(kwargs.get("data_filepath"))
    sample_names = data["sample_name"].unique().tolist()
    X = data["score"].values
    S = np.zeros((X.shape[0], len(sample_names)), dtype=bool)
    for i, sample_name in enumerate(sample_names):
        S[:, i] = data["sample_name"] == sample_name
    return X, S, sample_names

def run(**kwargs):
    """
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

    Returns:
    --------------------------
    best_fit -- Tuple[Tuple[float], Ndarray, Ndarray]
        The component parameters, weights, and likelihoods of the best fit
    """
    X, S, sample_names = load_data(**kwargs)
    bootstrap_indices = [tuple(range(si)) for si in S.sum(0)]
    if kwargs.get("bootstrap",True):
        X,S,bootstrap_indices = bootstrap(X,S,**kwargs)
    NUM_FITS = kwargs.get("num_fits", 25)
    save_path = kwargs.get("save_path", None)
    best_fit = []
    best_likelihood = -np.inf
    fit_results = Parallel(n_jobs=kwargs.get('core_limit',-1))(delayed(singleFit)(X, S, **kwargs) for i in range(NUM_FITS))
    for component_params, weights, likelihoods in fit_results:
        if likelihoods[-1] > best_likelihood:
            best_fit = (component_params, weights, likelihoods)
            best_likelihood = likelihoods[-1]
    if np.isinf(best_likelihood):
        print("No fits succeeded")
        return
    if save_path is not None:
        save(sample_names, best_fit, bootstrap_indices,**kwargs)
    return best_fit

if __name__ == "__main__":
    Fire(run)
