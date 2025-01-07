from typing import List,Optional,Dict
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import skewnorm, norm
from scipy.special import logsumexp

class MulticomponentCalibrationModel:
    """
    Multi-component skew-normal calibration model.

    Represent functionally normal (FN) and functionally abnormal (FA) distributions each as mixtures of skew normal distributions.

    Monotonicity is enforced between each pair of neighboring FN and FA components.

    Pathogenic, benign, and gnomAD assay-score distributions are modeled as mixtures of FN and FA mixture-distributions.
    """

    def __init__(self, component_classes : List[int],**kwargs):
        """
        Initialize the model with the given component classes.

        Parameters
        ----------
        component_classes : List[int]
            List of component classes, where each component class is either 0 (functionally normal) or 1 (functionally abnormal).
        """
        self.component_classes = component_classes
        self.num_components = len(component_classes)
        # get the index all functionally-abnormal and functionally normal components
        self.abnormal_indices = np.where(np.array(component_classes) == 1)[0]
        self.normal_indices = np.where(np.array(component_classes) == 0)[0]

    def fit(self, scores, sampleIndicators,**kwargs):
        """
        Fit the model to the given assay scores and sample indicators.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]
        """
        # Validate input data
        self.validate_inputs(scores, sampleIndicators)
        # Initialize model parameters (i.e., skewness, locs, scales, sample_weights)
        self.initialize_parameters(scores, sampleIndicators, **kwargs)
        # run the EM algorithm to fit the model to the given assay scores and sample indicators
        self._fit(scores, sampleIndicators, **kwargs)

    def _fit(self, scores, sampleIndicators, max_iter=100, tol=1e-6, **kwargs):
        """
        Run the EM algorithm to fit the model to the given assay scores and sample indicators.
        """
        
        # step 1) make sure monotonicity is enforced
        if self.components_violate_monotonicity(np.unique(scores)):
            raise ValueError("Initial model parameters violate monotonicity.")
        # step 2) update the parameters of each component
        for component_num in range(self.num_components):
            self._update_component_parameters(scores, sampleIndicators, component_num)
        # step 3) update mixture weights for each sample
        self._update_sample_weights(scores, sampleIndicators)
        raise NotImplementedError("Subclasses must implement the _fit method.")

    def _update_component_parameters(self, scores, sampleIndicators, component_num, **kwargs) -> None:
        """
        Update the parameters of the given component.
        
        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]
        component_num : int
            Component number.
        
        Returns
        -------
        None
        """
        self._update_component_location(scores, sampleIndicators, component_num)
        self._update_component_scale(scores, sampleIndicators, component_num)
        self._update_component_skewness(scores, sampleIndicators, component_num)
        raise NotImplementedError("Subclasses must implement the _update_component_parameters method.")

    @classmethod
    def get_truncated_normal_moments(cls,observations, component_params):
        _delta = cls._get_delta(component_params)
        loc, scale = component_params[1:]
        truncated_normal_loc = _delta / scale * (observations - loc)
        truncated_normal_scale = np.sqrt(1 - _delta**2)
        v, w = cls.trunc_norm_moments(truncated_normal_loc, truncated_normal_scale)
        return v, w

    @classmethod
    def trunc_norm_moments(cls, mu, sigma):
        """first and second truncated normal moments"""
        cdf = norm.cdf(mu / sigma)
        flags = cdf == 0
        pdf = norm.pdf(mu / sigma)
        p = np.zeros_like(pdf)
        p[~flags] = pdf[~flags] / cdf[~flags]
        p[flags] = abs(mu[flags] / sigma)
        m1 = mu + sigma * p
        m2 = mu**2 + sigma**2 + sigma * mu * p
        return m1, m2
    
    @classmethod
    def _get_delta(cls, skewness):
        return skewness / np.sqrt(1 + skewness**2)

    def validate_inputs(self, scores, sampleIndicators):
        nscores = scores.shape[0]
        nindicators,nsamples = sampleIndicators.shape
        assert nscores == nsamples, f"The number of scores ({nscores})must match the number of samples ({nindicators})."
        assert np.all(np.sum(sampleIndicators,axis=1) == 1), "sampleIndicators is expected to be a one-hot matrix."
        assert np.all(np.sum(sampleIndicators,axis=0) > 0), "each sample must have at least one observation."

    def initialize_parameters(self, scores, sampleIndicators,
                                skew_directions : Optional[List[int]]=None,
                                max_skew_init_magnitude=1, **kwargs) -> None:
        """
        Initialize the model parameters.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Optional Parameters
        -------------------
        - skew-directions : List[int] | None
            List of skew directions for each component, where each skew direction is either 1 (right-skewed), 0 (standard-normal), or -1 (left-skewed).
            If None, randomly assign skew directions to each component.
        - max-skew-init-magnitude : float | int (default 1)
            Maximum magnitude of the skew parameter for each skew-normal component.

        - see sklearn.cluster.KMeans for K-Means related parameters

        Returns
        -------
        None
        """
        # 1) Fit a k-means model to all assay scores
        self.kmeans_model = KMeans(n_clusters=self.num_components, **kwargs)
        scores = scores.reshape((-1, 1))
        component_assignments = self.kmeans_model.fit_predict(scores)
        # 2) Initialize skew-normal component parameters
        self._initialize_skew_normal_parameters(scores, component_assignments, skew_directions, max_skew_init_magnitude, **kwargs)
        # 3) Initialize the mixture weights
        self._initialize_sample_weights(scores, sampleIndicators)
        # 4) adjust the skew-normal component parameters to enforce monotonicity between FA and FN components
        self.adjust_to_monotonicity(np.unique(scores))

    def _initialize_skew_normal_parameters(self, scores : np.array,
                                            component_assignments : np.array,
                                            skew_directions : Optional[List[int]],
                                            max_skew_init_magnitude : float|int,
                                            **kwargs) -> None:
        """
        Initialize the skew-normal component parameters.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - component_assignments : numpy.array
            Component assignments (from k-means) for each assay score.
        - skew-directions : List[int] | None
            List of skew directions for each component, where each skew direction is either 1 (right-skewed), 0 (standard-normal), or -1 (left-skewed).
            If None, randomly assign skew directions to each component.
        - max-skew-init-magnitude : float | int
            Maximum magnitude of the skew parameter for each skew-normal component.

        Returns
        -------
        None
        """
        # initialize the skew-normal component parameters
        if skew_directions is None:
            skew_directions = np.random.choice([-1, 0, 1], self.num_components)
        else:
            assert len(skew_directions) == self.num_components, "The number of skew directions must match the number of components."
            assert all(skew_direction in [-1, 0, 1] for skew_direction in skew_directions), "Skew directions must be either -1 (left-skewed), 0 (standard-normal), or 1 (right-skewed)."
        self.skewness = np.zeros(self.num_components)
        self.locs = np.zeros(self.num_components)
        self.scales = np.zeros(self.num_components)
        indexMapping = self._groupValues(component_assignments)
        for componentNum,skewDirection in enumerate(skew_directions):
            if skewDirection == 1:
                self.skewness[componentNum] = np.random.uniform(0, max_skew_init_magnitude)
            elif skewDirection == -1:
                self.skewness[componentNum] = np.random.uniform(-max_skew_init_magnitude, 0)
            component_scores = scores[indexMapping[componentNum]]
            self.locs[componentNum] = np.mean(component_scores)
            self.scales[componentNum] = np.std(component_scores)

    def adjust_to_monotonicity(self, scores : np.array, **kwargs) -> None:
        """
        Adjust the skew-normal component parameters to enforce monotonicity between the FN and FA components.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        
        Optional Parameters
        -------------------
        - max_monotonicity_reduction_iters : int (default 100)
            Maximum number of iterations to reduce the skewness and scales parameters to enforce monotonicity.

        Returns
        -------
        List[numpy.array]
            Adjusted skewness, locs, and scales parameters.
        """
        reduction_iters = 0
        while self.components_violate_monotonicity(scores) and reduction_iters < kwargs.get("max_monotonicity_reduction_iters", 100):
            reduction_iters += 1
            for i in range(self.num_components - 1):
                self.skewness[i] *= 0.95
                self.scales[i] *= 0.95
                self.skewness[i + 1] *= 0.95
                self.scales[i + 1] *= 0.95
        if self.components_violate_monotonicity(scores):
            raise ValueError("Could not enforce monotonicity between components.")

    def components_violate_monotonicity(self, scores, **kwargs) -> bool:
        """
        Check whether the joint density ratio of the FA to FN components is monotonic
        
        Parameters
        ----------
        - scores : numpy.array
            Assay scores.

        Returns
        -------
        bool
            True if the joint density ratio of the FA to FN components is non-monotonic, False otherwise.
        """
        # pathogenic distribution pdf first derivative
        p_prime = self._mixture_pdf_first_derivative(scores, self.sample_weights[0])
        # benign distribution pdf first derivative
        b_prime = self._mixture_pdf_first_derivative(scores, self.sample_weights[1])
        # pathogenic distribution pdf
        p = np.sum(self.sample_weights[0][...,None] * [self.get_component_density(scores, i) for i in range(self.num_components)], axis=0)
        # benign distribution pdf
        b = np.sum(self.sample_weights[1][...,None] * [self.get_component_density(scores, i) for i in range(self.num_components)], axis=0)
        # log density ratio first derivative (numerator only, as denominator is always positive)
        ldr_prime = p_prime * b - b_prime * p # denominator = b**2
        return not ((ldr_prime > 0).all() or (ldr_prime < 0).all())

    def _mixture_pdf_first_derivative(self, scores : np.array, weights : np.array, **kwargs) -> np.array:
        """
        Get the first derivative of the mixture density.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - weights : numpy.array
            Weight of each component for the given mixture

        Returns
        -------
        numpy.array
            First derivative of the mixture density.
        """
        def _comp_derivitive(scores, componentNum):
            skew, loc, scale = self.skewness[componentNum], self.locs[componentNum], self.scales[componentNum]
            return (2 * skew / scale**2) * \
                norm.pdf((scores - loc) / scale) * \
                    norm.pdf((skew / scale) * (scores - loc))
        return np.sum([weights[i] * _comp_derivitive(scores, i) for i in range(self.num_components)], axis=0)


    def _groupValues(self, vals):
        groups = {}
        for i, val in enumerate(vals):
            if val not in groups:
                groups[val] = []
            groups[val].append(i)
        return groups

    def _groups_from_onehot(self, sampleIndicators : np.array) -> Dict[int, List[int]]:
        """
        Convert one-hot sample indicators to a dictionary of column indices and row indices.
        """
        # Initialize a dictionary to store the column index and list of row indices
        indices_dict = {}

        # Iterate through the sampleIndicators
        for row_idx, row in enumerate(sampleIndicators):
            for col_idx, value in enumerate(row):
                if value == 1:
                    if col_idx not in indices_dict:
                        indices_dict[col_idx] = []
                    indices_dict[col_idx].append(row_idx)
                    break
        return indices_dict

    def _initialize_sample_weights(self, scores : np.array, sampleIndicators : np.array) -> None:
        """
        Initialize `sample_weights`.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]
        

        Returns
        -------
        None
        """
        component_posteriors = self.get_component_posteriors(scores, np.ones(self.num_components) / self.num_components)
        self.sample_weights = np.zeros((self.sampleIndicators.shape[1], self.num_components))
        sample_to_indices = self._groups_from_onehot(sampleIndicators)
        for sampleIdx, indices in sample_to_indices.items():
            self.sample_weights[sampleIdx] = np.mean(component_posteriors[indices], axis=0)

    def get_component_density(self, scores : np.array, component_num : int, **kwargs) -> np.array:
        """
        Get the density of the scores.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - component_num : int
            Component number.

        Optional Parameters
        -------------------
        - log : bool (default False)
            If True, return the log-density of the  scores.
        Returns
        -------
        numpy.array
            Density of the scores.
        """
        if kwargs.get("log", False):
            return skewnorm.logpdf(scores, self.skewness[component_num], self.locs[component_num], self.scales[component_num])
        return skewnorm.pdf(scores, self.skewness[component_num], self.locs[component_num], self.scales[component_num])
    
    def get_component_posteriors(self, scores : np.array, weights,**kwargs) -> np.array:
        """
        Get the posterior probabilities of each scores.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - weights : numpy.array
            Weight of each component for the given mixture

        Optional Parameters
        -------------------
        - logsumpexp : bool (default False)
            If True, use logsumexp to compute the posterior probabilities.
        Returns
        -------
        numpy.array
            Posterior probabilities of each scores.
        """
        if len(weights) != self.num_components:
            raise ValueError(f"The number of sample weights {len(weights)} must match the number of components {self.num_components}.")
        weights = np.array(weights)[:, None]
        comp_posteriors = np.zeros((scores.shape[0], self.num_components))
        if kwargs.get("logsumpexp", True):
            log_pdfs = np.stack([self.get_component_density(scores, i, log=True) for i in range(self.num_components)], axis=0)
            numerators = np.zeros_like(log_pdfs)
            numerators = log_pdfs + np.log(weights)
            d = logsumexp(numerators, axis=0)
            comp_posteriors = np.exp(numerators - d[None])
            comp_posteriors[np.isnan(comp_posteriors)] = 0
        else:
            for componentNum in range(self.num_components):
                comp_posteriors[:, componentNum] = self.get_component_density(scores, componentNum) * weights[componentNum]
            comp_posteriors /= np.sum(comp_posteriors, axis=1)[:, np.newaxis]
        return comp_posteriors

    def predict(self, scores, weights,**kwargs):
        """
        Predict the probability of each component for each score

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        weights : numpy.array
            Weight of each component for the given mixture

        Returns
        -------
        numpy.array
            Predicted probability of each component for each score
        """
        return self.get_component_posteriors(scores, weights,**kwargs)

    def fit_predict(self, scores, sampleIndicators,**kwargs):
        """
        Fit the model to the given assay scores and sample indicators, and predict the probability of each sample being pathogenic, benign, or gnomAD.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., [benign, pathogenic, gnomAD, synonymous]

        Returns
        -------
        numpy.array
            Predicted probabilities of each sample being pathogenic, benign, or gnomAD.
        """
        # fit the model to the given assay scores and sample indicators
        # predict the probability of each sample being pathogenic, benign, or gnomAD
        self.fit(scores, sampleIndicators)
        return self.predict(scores)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameters for this estimator.
        """
        raise NotImplementedError("Subclasses must implement the get_params method.")

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Returns
        -------
        self
        """
        raise NotImplementedError("Subclasses must implement the set_params method.")

    def get_metadata_routing(self) -> None:
        """
        For compatibility with scikit-learn models

        Returns
        -------
        None
        """
        pass