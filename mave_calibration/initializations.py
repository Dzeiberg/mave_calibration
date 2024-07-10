from mave_calibration.skew_normal.fit import fit_skew_normal
from mave_calibration.skew_normal import density_utils
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np

def gmm_init(X,**kwargs):
    """
    Initialize the parameters of the skew normal mixture model using GMM and the skew-normal method of moments
    """
    n_components = kwargs.get("n_components",2)
    n_inits = kwargs.get("n_inits",10)
    gmm = GaussianMixture(n_components=n_components)
    X = np.array(X).reshape((-1,1))
    gmm.fit(X)
    gmm_params = sorted([(0, loc, scale) for loc, scale in zip(gmm.means_.ravel(), np.sqrt(gmm.covariances_).ravel())],
                            key=lambda tup: tup[1])
    component_responsibilities = density_utils.component_posteriors(X, gmm_params, gmm.weights_)
    best_parameters = []
    best_weights = np.zeros(n_components)
    BestLL = -1e10
    for rep in range(n_inits):
        component_parameters = []
        comp_weights = np.zeros(n_components)
        for i in range(n_components):
            comp_mask = np.random.binomial(1, component_responsibilities[i]).astype(bool)
            comp_weights[i] = comp_mask.sum() / len(X)
            X_component = X[comp_mask]
            params = fit_skew_normal(X_component)
            component_parameters.append(params)
        LL = np.log(density_utils.mixture_pdf(X, component_parameters, comp_weights)).sum() / len(X)
        if LL > BestLL:
            best_parameters = component_parameters
            best_weights = comp_weights
            BestLL = LL

    return best_parameters



def kmeans_init(X,**kwargs):
    """
    Initialize the parameters of the skew normal mixture model using kmeans and the method of moments
    """
    n_clusters = kwargs.get("n_clusters",2)
    init = kwargs.get("kmeans_init",'random')
    kmeans = KMeans(n_clusters=n_clusters,init=init)

    X = np.array(X).reshape((-1,1))

    # cluster_assignments = kmeans.fit_predict(X)
    kmeans.fit(X)
    kmeans.cluster_centers_ = np.sort(kmeans.cluster_centers_,axis=0)
    cluster_assignments = kmeans.predict(X)
    component_weights = np.bincount(cluster_assignments) / len(X)

    cluster_centers = kmeans.cluster_centers_.ravel()

    component_parameters = []
    for i in range(n_clusters):
        X_cluster = X[cluster_assignments == i]
        params = fit_skew_normal(X_cluster)
        component_parameters.append(params)
    return component_parameters