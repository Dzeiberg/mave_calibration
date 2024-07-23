from mave_calibration.skew_normal.fit import fit_skew_normal
from mave_calibration.skew_normal import density_utils
from mave_calibration.em_opt.constraints import density_constraint_violated
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from tqdm import trange

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
    assert n_components == 2
    n_inits = kwargs.get("n_inits", 10)
    X = np.array(X).reshape((-1, 1))
    gmm_component_responsibilities = get_gmm_responsibilities(X, **kwargs)
    best_parameters = []
    BestLL = -1e10
    if kwargs.get('verbose',True):
        _range = trange(n_inits,leave=False)
    else:
        _range = range(n_inits)
    for rep in _range:
        component_parameters = []
        comp_weights = np.zeros(n_components)
        for i in range(n_components):
            X_component = sample_from_gmm(X, gmm_component_responsibilities[i])
            comp_weights[i] = len(X_component) / len(X)

            params = fit_skew_normal(X_component)
            component_parameters.append(params)
        for _ in range(10000):
            if not density_constraint_violated(
                component_parameters[0], component_parameters[1], (X.min(), X.max())
            ):
                break
            larger_comp_index = np.argmax([p[2] for p in component_parameters])
            component_parameters[larger_comp_index] = [component_parameters[larger_comp_index][0] * .99,
                                                        component_parameters[larger_comp_index][1],
                                                        component_parameters[larger_comp_index][2] * .99]
        if density_constraint_violated(
            component_parameters[0], component_parameters[1], (X.min(), X.max())
        ):
            print(f"init {i} failed")
            continue
        LL = get_LL(X, component_parameters, comp_weights)
        if LL > BestLL:
            best_parameters = component_parameters
            BestLL = LL
    if not len(best_parameters):
        raise ValueError("Could not initialize to satisfy density constraint")
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
