import scipy.stats as sps
import numpy as np
from multicomp_calibration.multicomp_model import MulticomponentCalibrationModel

def generate_data():
    skewness = [0.1, 0.2, 0.3]
    loc = [0, 1, 2]
    scale = [1, 1.5, 1]
    sample_weights = [[0.95, 0.05, 0],
                      [0, 0.25, 0.75],
                      [0.5, 0.5, 0],
                      [0, 0, 1]]
    n_samples = len(sample_weights)
    n_components = len(skewness)
    observations = []
    sampleNums = []
    ObservationsPerSample = 1000
    for sampleNum in range(n_samples):
        for componentNum in range(n_components):
            n = int(ObservationsPerSample * sample_weights[sampleNum][componentNum])
            observations.extend(sps.skewnorm.rvs(skewness[componentNum], loc[componentNum], scale[componentNum], size=n))
            sampleNums.extend([sampleNum] * n)
    sampleIndicators = np.zeros((len(observations),n_samples))
    sampleIndicators[np.arange(len(observations)),sampleNums] = 1
    trueParameters = dict(skewness=skewness, loc=loc, scale=scale, sample_weights=sample_weights)
    return np.array(observations), sampleIndicators, trueParameters

def test_multicomp():
    scores, sampleIndicators, trueParameters = generate_data()
    model = MulticomponentCalibrationModel(3)
    model.fit(scores, sampleIndicators,check_convergence=False)
    for sampleNum in range(sampleIndicators.shape[1]):
        cdfdist = model.get_cdf_distance(scores[:,sampleNum], sampleNum)
        print(f"Sample {sampleNum} CDF distance: {cdfdist}")
    assert np.allclose(model.skewness, trueParameters['skewness'], atol=0.1)
    assert np.allclose(model.loc, trueParameters['loc'], atol=0.1)
    assert np.allclose(model.scale, trueParameters['scale'], atol=0.1)
    assert np.allclose(model.sample_weights, trueParameters['sample_weights'], atol=0.1)

test_multicomp()