from mave_calibration.skew_normal.fit import fit_skew_normal
import numpy as np

def test_fit_on_vector():
    X=[-3.9516,-3.1752,-1.3199,-3.6848,-5.5401,-2.2428,-0.7689, 1]
    params = fit_skew_normal(X, method="mm")
    assert np.allclose(params, [0.2256,-3.1547,2.1803], atol=1e-4)

if __name__ == "__main__":
    test_fit_on_vector()