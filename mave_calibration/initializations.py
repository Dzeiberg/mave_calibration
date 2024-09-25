from mave_calibration.skew_normal.fit import fit_skew_normal
from mave_calibration.skew_normal import density_utils
from mave_calibration.em_opt.constraints import density_constraint_violated
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from tqdm import trange
import logging

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

def constrained_gmm_init(X, **kwargs):
    """
    Density Constrained initialization of the skew normal mixture model using GMM and the skew-normal method of moments
    """
    n_components = kwargs.get("n_components", 2)
    if n_components != 2:
        logging.warning(f"Density constraint only enforced for first two components; {n_components} components requested")
    n_inits = kwargs.get("n_inits", 100)
    X = np.array(X).reshape((-1, 1))
    buffer_stds = kwargs.get("buffer_stds", 0)
    obs_std = X.std()
    xlims = (X.min() - obs_std * buffer_stds,
             X.max() + obs_std * buffer_stds)
    gmm_component_responsibilities = get_gmm_responsibilities(X, **kwargs)
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
    for rep in _range:
        component_parameters = []
        comp_weights = np.zeros(n_components)
        for i in range(n_components):
            X_component = sample_from_gmm(X, gmm_component_responsibilities[i])
            comp_weights[i] = len(X_component) / len(X)

            params = fit_skew_normal(X_component, method=init_method[rep])
            component_parameters.append(params)
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
                    # component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0] * -.9,
                    #                                             component_parameters[larger_comp_index][1],
                    #                                             component_parameters[larger_comp_index][2]]
                    # ONLY Scale down the skew
                    component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0] * .9,
                                                                component_parameters[larger_comp_index][1],
                                                                component_parameters[larger_comp_index][2]]
                else:
                    # Scale down the scale
                    component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0],
                                                                component_parameters[larger_comp_index][1],
                                                                component_parameters[larger_comp_index][2] * .9]
            if density_constraint_violated(
                component_parameters[compI], component_parameters[compJ], xlims
            ):
                print(f"init {rep} failed; final parameters: {component_parameters[compI]}\t{component_parameters[compJ]}")
                rep_failed = True
                break
        if rep_failed:
            continue
        LL = get_LL(X, component_parameters, comp_weights)
        if LL > BestLL:
            best_parameters = component_parameters
            BestLL = LL
    if not len(best_parameters):
        print("Could not initialize to satisfy density constraint; defaulting to random")
        best_parameters = [[-.25, X.min(), 2],
                           [.25, X.max(), 2]]
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

def kmeans_init(X, **kwargs):
    """
    Initialize the parameters of the skew normal mixture model using kmeans and the method of moments
    """
    n_clusters = kwargs.get("n_clusters", 2)
    init = kwargs.get("kmeans_init", "random")
    kmeans = KMeans(n_clusters=n_clusters, init=init)

    X = np.array(X).reshape((-1, 1))

    # cluster_assignments = kmeans.fit_predict(X)
    kmeans.fit(X)
    kmeans.cluster_centers_ = np.sort(kmeans.cluster_centers_, axis=0)
    cluster_assignments = kmeans.predict(X)

    component_parameters = []
    for i in range(n_clusters):
        X_cluster = X[cluster_assignments == i]
        params = fit_skew_normal(X_cluster)
        component_parameters.append(params)
    return component_parameters
