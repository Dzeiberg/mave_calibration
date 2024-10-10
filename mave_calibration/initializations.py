from mave_calibration.skew_normal.fit import fit_skew_normal
from mave_calibration.skew_normal import density_utils
from mave_calibration.em_opt.constraints import density_constraint_violated
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from tqdm import trange
import logging
import scipy.stats as sps

from scipy.stats._warnings_errors import FitError

def fit_gmm(X,**kwargs):
    """
    Fit a Gaussian Mixture Model to the data
    """
    n_components = kwargs.get("n_components", 2)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    gmm_params = sorted(
        [
            ((0, loc, scale),weight)
            for loc, scale,weight in zip(gmm.means_.ravel(), np.sqrt(gmm.covariances_).ravel(),gmm.weights_)
        ],
        key=lambda tup: tup[0][1],
    )
    gmm_weights = [t[1] for t in gmm_params]
    gmm_params = [t[0] for t in gmm_params]
    return gmm_params, gmm_weights

def get_gmm_responsibilities(X, **kwargs):
    """
    Get gmm responsibilities for the data
    """
    gmm_params, gmm_weights = fit_gmm(X, **kwargs)
    component_responsibilities = density_utils.component_posteriors(
        X, gmm_params, gmm_weights
    )
    return component_responsibilities


def sample_from_gmm(X, component_responsibilities):
    """
    Sample from a Gaussian Mixture Model

    Arguments:
    X: np.array (N,): observed instances
    component_responsibilities: np.array (N,): posterior probabilities of each component given x

    Returns:
    X_component: np.array (M,) | M<=N: sampled instances from the component
    """
    comp_mask = np.random.binomial(1, component_responsibilities).astype(bool)
    X_component = X[comp_mask]
    return X_component

def constrained_gmm_init(observations, sample_indicators,**kwargs):
    """
    Density Constrained initialization of the skew normal mixture model using GMM and the skew-normal method of moments
    """
    n_components = kwargs.get("n_components", 2)
    if n_components != 2:
        logging.warning(f"Density constraint only enforced for first two components; {n_components} components requested")
    n_inits = kwargs.get("n_inits", 100)
    best_parameters = []
    BestLL = -1e10
    if kwargs.get('verbose',True):
        _range = trange(n_inits,leave=False)
    else:
        _range = range(n_inits)
    if kwargs.get('skewnorm_init_method',None) is None:
        init_method = ['mle' if i % 2 == 0 else 'mm' for i in range(n_inits)]
    else:
        init_method = [kwargs.get('skewnorm_init_method') for _ in range(n_inits)]
    all_observations = np.concatenate(observations)
    xlims = (all_observations.min(),all_observations.max())
    for rep in _range:
        Nsamples = sample_indicators.shape[1]
        sampleNumToUse = np.random.randint(0,Nsamples+1)
        if sampleNumToUse == Nsamples:
            X = np.concatenate(observations)
        else:
            # X = observations[sample_indicators[:,sampleNumToUse]]
            X = np.concatenate([np.array(obs) for obs,ind in zip(observations, sample_indicators[:,sampleNumToUse]) if ind])
        X = X[np.random.randint(0,len(X),size=len(X))].reshape((-1, 1))
        if len(X) < 2:
            continue
        gmm_component_responsibilities = get_gmm_responsibilities(X, **kwargs)
        component_parameters = []
        comp_weights = np.zeros(n_components)
        failed_init_component = False
        for i in range(n_components):
            X_component = sample_from_gmm(X, gmm_component_responsibilities[i])
            comp_weights[i] = len(X_component) / len(X)
            try:
                params = fit_skew_normal(X_component, method=init_method[rep])
            except FitError:
                failed_init_component = True
                break
            component_parameters.append(params)
        if failed_init_component:
            continue
        rep_failed = False
        for compI, compJ in zip(range(0,n_components-1),range(1,n_components)):
            for _ in range(300):
                if not density_constraint_violated(
                    component_parameters[compI], component_parameters[compJ], xlims
                ):
                    break
                # identify whether compI or compJ has the larger magnitude of skew or scale
                larger_comp_index = compI
                magnitudeI = max(abs(component_parameters[compI][0]),component_parameters[compI][2])
                magnitudeJ = max(abs(component_parameters[compJ][0]),component_parameters[compJ][2])
                # magnitudeI = component_parameters[compI][2]
                # magnitudeJ = component_parameters[compJ][2]
                if magnitudeJ > magnitudeI:
                    larger_comp_index = compJ
                if abs(component_parameters[larger_comp_index][0]) > component_parameters[larger_comp_index][2]:
                # if False:
                    # Scale down and flip the skew
                    component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0] * -.9,
                                                                component_parameters[larger_comp_index][1],
                                                                component_parameters[larger_comp_index][2]]
                    # ONLY Scale down the skew
                    # component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0] * .9,
                    #                                             component_parameters[larger_comp_index][1],
                    #                                             component_parameters[larger_comp_index][2]]
                else:
                    # Scale down the scale
                    component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0],
                                                                component_parameters[larger_comp_index][1],
                                                                component_parameters[larger_comp_index][2] * .9]
            if density_constraint_violated(
                component_parameters[compI], component_parameters[compJ], xlims
            ):
                # if kwargs.get('verbose',True):
                #     print(f"init {rep} failed; final parameters: {component_parameters[compI]}\t{component_parameters[compJ]}")
                rep_failed = True
                break
        if rep_failed:
            continue
        assert not density_constraint_violated(
            component_parameters[0], component_parameters[1], xlims
        )
        LL = get_LL(X, component_parameters, comp_weights)
        if LL > BestLL:
            best_parameters = component_parameters
            if kwargs.get('verbose',True):
                print(f"init {rep} succeeded; new best LL: {LL}\n{component_parameters[0]}{component_parameters[1]}")
            BestLL = LL
    if not len(best_parameters):
        print("Could not initialize to satisfy density constraint; defaulting to random")
        best_parameters = [[-.25, X.min(), 2],
                           [.25, X.max(), 2]]

    if kwargs.get('verbose', True):
        print("FINAL INITIALIZED PARAMETERS: \n{}\n{}".format(best_parameters[0],best_parameters[1]))
    return best_parameters
    

def gmm_init(X, **kwargs):
    """
    Initialize the parameters of the skew normal mixture model using GMM and the skew-normal method of moments
    """
    n_components = kwargs.get("n_components", 2)
    n_inits = kwargs.get("n_inits", 10)
    X = np.array(X).reshape((-1, 1))
    gmm_component_responsibilities = get_gmm_responsibilities(X, **kwargs)
    best_parameters = []
    BestLL = -1e10
    for rep in range(n_inits):
        component_parameters = []
        comp_weights = np.zeros(n_components)
        for i in range(n_components):
            X_component = sample_from_gmm(X, gmm_component_responsibilities[i])
            comp_weights[i] = len(X_component) / len(X)

            params = fit_skew_normal(X_component)
            component_parameters.append(params)
        LL = get_LL(X, component_parameters, comp_weights)
        if LL > BestLL:
            best_parameters = component_parameters
            BestLL = LL

    return best_parameters

def get_LL(X, component_parameters, comp_weights):
    """
    Calculate the log likelihood of the data given the parameters of the skew normal mixture model
    """
    return np.log(density_utils.mixture_pdf(X, component_parameters, comp_weights)).sum() / len(X)

def fix_to_satisfy_density_constraint(component_parameters, xlims):
    n_components = len(component_parameters)
    rep_failed = False
    for compI, compJ in zip(range(0,n_components-1),range(1,n_components)):
        if rep_failed:
            break
        for _ in range(300):
            if not density_constraint_violated(
                component_parameters[compI], component_parameters[compJ], xlims
            ):
                break
            component_parameters[compI] = [component_parameters[compI][0] - .05 * abs(component_parameters[compI][0]),
                                           component_parameters[compI][1],
                                           component_parameters[compI][2]]
            component_parameters[compJ] = [component_parameters[compJ][0] + .05 * abs(component_parameters[compJ][0]),
                                           component_parameters[compJ][1],
                                           component_parameters[compJ][2]]

        if density_constraint_violated(
            component_parameters[compI], component_parameters[compJ], xlims
        ):
            rep_failed = True
            break
    if rep_failed:
        return [[] for _ in range(n_components)]
    assert not density_constraint_violated(
        component_parameters[0], component_parameters[1], xlims
    )
    return component_parameters

def kmeans_init(X, **kwargs):
    """
    Initialize the parameters of the skew normal mixture model using kmeans and the method of moments

    Arguments:
    X: np.array (N,): observed instances

    Optional Keyword Arguments:
    - n_clusters: int: number of clusters to use in kmeans. Default: 2
    - kmeans_init: str: initialization method for kmeans. Options: ["random", "k-means++"]. Default: "random"
    - skewnorm_init_method: str: method to use for fitting the skew normal distribution. Options: ["mle", "mm"]. Default: "mle"
    """
    repeat = 0
    while repeat < 1000:
        n_clusters = kwargs.get("n_clusters", 2)
        init = kwargs.get("kmeans_init", "random")
        kmeans = KMeans(n_clusters=n_clusters, init=init)

        X = np.array(X).reshape((-1, 1))
        kmeans.fit(X)
        cluster_assignments = kmeans.predict(X)

        component_parameters = []
        for i in range(n_clusters):
            X_cluster = X[cluster_assignments == i]
            loc,scale = sps.norm.fit(X_cluster)
            a = np.random.uniform(-.25,.25)
            component_parameters.append((a,float(loc),float(scale)))
        component_parameters = fix_to_satisfy_density_constraint(component_parameters,(X.min(),X.max()))
        if not len(component_parameters[0]):
            repeat += 1
        else:
            return component_parameters, kmeans
    raise ValueError("Failed to initialize")

def random_init(X, **kwargs):
    """
    Initialize the parameters of the skew normal mixture model using random initialization

    Arguments:
    X: np.array (N,): observed instances

    Optional Keyword Arguments:
    - n_clusters: int: number of clusters to use in kmeans. Default: 2
    - skewnorm_init_method: str: method to use for fitting the skew normal distribution. Options: ["mle", "mm"]. Default: "mle"
    """
    repeat = 0
    while repeat < 1000:
        n_clusters = kwargs.get("n_clusters", 2)
        component_parameters = []
        for i in range(n_clusters):
            loc = np.random.uniform(X.min(), X.max())
            scale = np.random.uniform(1e-5, X.max() - X.min())
            a = np.random.uniform(kwargs.get('skew_min',-5),kwargs.get('skew_max',5))
            component_parameters.append((a,loc,scale))
        component_parameters = fix_to_satisfy_density_constraint(component_parameters,(X.min(),X.max()))
        if not len(component_parameters[0]):
            repeat += 1
        else:
            return component_parameters
    raise ValueError("Failed to initialize")