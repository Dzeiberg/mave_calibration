import sys
sys.path.append("/home/dz/research/mave_calibration")
from mave_calibration.skew_normal.fit import fit_skew_normal
import numpy as np
import scipy.stats as sps

def test_fit_on_vector():
    X=[-3.9516,-3.1752,-1.3199,-3.6848,-5.5401,-2.2428,-0.7689, 1]
    params = fit_skew_normal(X)
    assert np.allclose(params, [0.2256,-3.1547,2.1803], atol=1e-4)

# def test_fit_skew_normal_right():
#     # Generate some data
#     true_params = [2, 0, 1]
#     data = sps.skewnorm.rvs(a=true_params[0],loc=true_params[1],scale=true_params[2], size=100)
#     params = fit_skew_normal(data)

#     assert np.allclose(params, true_params, atol=1e-4)

# def test_fit_skew_normal_left():
#     # Generate some data
#     true_params = [-1, 0, 1]
#     data = sps.skewnorm.rvs(a=true_params[0],loc=true_params[1],scale=true_params[2], size=100)
#     params = fit_skew_normal(data)

#     assert np.allclose(params, true_params, atol=1e-4)

if __name__ == "__main__":
    test_fit_on_vector()
    print("All tests passed.")