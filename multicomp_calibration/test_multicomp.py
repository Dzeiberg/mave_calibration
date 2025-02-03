import scipy.stats as sps
import numpy as np
from multicomp_calibration.multicomp_model import MulticomponentCalibrationModel
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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

def try_fit(scores, sampleIndicators, trueParameters):
    try:
        model = MulticomponentCalibrationModel(3)
        model.fit(scores, sampleIndicators)
        return model
    except Exception as e:
        model._log_likelihoods.append(-np.inf)
        return model

def test_multicomp():
    scores, sampleIndicators, trueParameters = generate_data()
    fits = Parallel(n_jobs=4)(delayed(try_fit)(scores, sampleIndicators, trueParameters) for _ in range(4))
    fits = sorted(fits, key=lambda x: x._log_likelihoods[-1], reverse=True)
    # model = MulticomponentCalibrationModel(3)
    # model.fit(scores, sampleIndicators,check_convergence=True)
    model = fits[0]
    cdfdists = []
    for sampleNum in range(sampleIndicators.shape[1]):
        uscores = sorted(scores[sampleIndicators[:,sampleNum] == 1])
        empirical_cdf = model.empirical_cdf(uscores)
        model_cdf = model.get_sample_cdf(uscores, sampleNum)
        cdfdist = model.yang_dist(empirical_cdf, model_cdf)
        fig,ax = plt.subplots(1,1)
        ax.plot(uscores, empirical_cdf, label='Empirical CDF')
        ax.plot(uscores, model_cdf, label='Model CDF')
        ax.legend()
        ax.set_title(f"Sample {sampleNum} CDF distance: {cdfdist}")
        fig.savefig(".pytest/multicomp_test_sample{}.png".format(sampleNum))
        print(f"Sample {sampleNum} CDF distance: {cdfdist}")
        cdfdists.append(cdfdist)
    
    # assert np.allclose(model.skewness, trueParameters['skewness'], atol=0.1)
    # assert np.allclose(model.loc, trueParameters['loc'], atol=0.1)
    # assert np.allclose(model.scale, trueParameters['scale'], atol=0.1)
    # assert np.allclose(model.sample_weights, trueParameters['sample_weights'], atol=0.1)
    assert np.all(np.array(cdfdists) < 0.1)

test_multicomp()