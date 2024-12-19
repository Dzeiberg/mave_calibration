from typing import List,Optional,Dict
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import skewnorm

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
        # get the index pairs of all sign changes (i.e., the index of the FN (FA) component that precedes the FA (FN) component)
        self.sign_change_indices = [i for i in range(self.num_components - 1) if component_classes[i] != component_classes[i + 1]]

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
        pass

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
        self.skewness, self.locs, self.scales = self.adjust_to_monotonicity(scores, self.skewness, self.locs, self.scales)

    def adjust_to_monotonicity(self, scores, skewness : np.array, locs : np.array, scales : np.array) -> List[np.array]:
        """
        Adjust the skew-normal component parameters to enforce monotonicity between each pair of neighboring FN and FA components.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - skewness : numpy.array
            Skewness parameters for each component.
        - locs : numpy.array
            Location parameters for each component.
        - scales : numpy.array
            Scale parameters for each component.

        Returns
        -------
        List[numpy.array]
            Adjusted skewness, locs, and scales parameters.
        """
        raise NotImplementedError()
        
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
        component_posteriors = self.get_component_posteriors(scores)
        self.sample_weights = np.zeros((self.sampleIndicators.shape[1], self.num_components))
        sample_to_indices = self._groups_from_onehot(sampleIndicators)
        for sampleIdx, indices in sample_to_indices.items():
            self.sample_weights[sampleIdx] = np.mean(component_posteriors[indices], axis=0)

    def get_component_posteriors(self, scores : np.array) -> np.array:
        """
        Get the posterior probabilities of each component given the assay scores.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.

        Returns
        -------
        numpy.array
            Posterior probabilities of each component given the assay scores.
        """
        comp_posteriors = np.zeros((scores.shape[0], self.num_components))
        for componentNum in range(self.num_components):
            comp_posteriors[:, componentNum] = skewnorm.pdf(scores, self.skewness[componentNum], self.locs[componentNum], self.scales[componentNum])
        comp_posteriors /= np.sum(comp_posteriors, axis=1)[:, np.newaxis]
        return comp_posteriors

    def predict(self, scores, sampleIndicators,**kwargs):
        """
        Predict the probability of each sample being pathogenic, benign, or gnomAD.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Returns
        -------
        numpy.array
            Predicted probabilities of each sample being pathogenic, benign, or gnomAD.
        """
        # predict the probability of each sample being pathogenic, benign, or gnomAD
        pass

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
        return self.predict(scores, sampleIndicators)

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
        pass

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Returns
        -------
        self
        """
        pass

    def get_metadata_routing(self) -> None:
        """
        For compatibility with scikit-learn models

        Returns
        -------
        None
        """
        pass