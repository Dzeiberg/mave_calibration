import numpy as np
import scipy.stats as sps

def mixture_pdf(x, params, weights):
    return np.sum([w * sps.skewnorm.pdf(x, a, loc, scale) for (a, loc, scale), w in zip(params, weights)], axis=0)