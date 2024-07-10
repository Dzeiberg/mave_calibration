from mave_calibration.skew_normal.fit import fit_skew_normal

from sklearn.cluster import KMeans
import numpy as np

def kmeans_init(X,**kwargs):
    """
    Initialize the parameters of the skew normal mixture model using kmeans and the method of moments
    """
    n_clusters = kwargs.get("n_clusters",2)
    init = kwargs.get("kmeans_init",'random')
    kmeans = KMeans(n_clusters=n_clusters,init=init)

    X = np.array(X).reshape((-1,1))

    cluster_assignments = kmeans.fit_predict(X)
    component_weights = np.bincount(cluster_assignments) / len(X)

    cluster_centers = kmeans.cluster_centers_.ravel()

    component_parameters = []
    for i in range(n_clusters):
        X_cluster = X[cluster_assignments == i]
        params = fit_skew_normal(X_cluster)
        component_parameters.append(params)
    return component_parameters, component_weights