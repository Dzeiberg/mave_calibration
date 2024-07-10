import scipy.stats as sps
import numpy as np

def fit_skew_normal(X):
    """
    Fit a skew normal distribution to the data.
    """
    return list(map(float,sps.skewnorm.fit(X)))
    
def method_of_moments(X):
    params = [0,0,0]

    m1 = np.mean(X)
    m2 = np.var(X,ddof=1)
    m3 = sps.skew(X)


    a1 = np.sqrt(2/np.pi)
    b1 = (4/np.pi - 1) / a1

    delta = np.sign(m3) / np.sqrt(a1**2 + m2 * (b1 / abs(m3))**(2/3))

    scale = np.sqrt(m2 / (1 - a1**2 * delta**2))

    loc = m1 - a1*delta*scale

    params = [m3.item(), loc.item(), scale.item()]
    return params

