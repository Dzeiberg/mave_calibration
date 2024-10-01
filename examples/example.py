#import sys
#sys.path.append("..")

from mave_calibration.plotting.generate_aggregate_figs import get_lrPlus, prior_from_weights,get_score_thresholds
from mave_calibration.main import singleFit, load_data

import os
from pathlib import Path
import pandas as pd
import numpy as np

def runExample():
    # location of this file
    pth = os.path.dirname(os.path.abspath(__file__))
    # dataset name
    dataset_name = "Findlay_BRCA1_SGE"
    # dataset directory
    dataset_dir = Path(os.path.join(pth,dataset_name))
    observations, sample_indicators, sample_names = load_data(data_filepath=dataset_dir / "samples.csv")
    # run the calibration
    bestFit = singleFit(observations, sample_indicators,
                   max_iters=10000, n_inits=100, verbose=True)
    score_range =np.arange(observations.min(), observations.max(), .01)
    lrPlus = get_lrPlus(score_range, sample_names.index('B/LB'), bestFit)
    prior = prior_from_weights(bestFit.weights,
                                    controls_idx=sample_names.index('B/LB'))
    pathogenic_score_thresholds, benign_score_thresholds = get_score_thresholds(lrPlus, prior, score_range)
    print("Pathogenic score thresholds:")
    for num_points, score_threshold in zip("+1 +2 +3 +4 +8".split(" "), pathogenic_score_thresholds):
        print(f"{num_points}: {'-------' if np.isnan(score_threshold) else np.round(score_threshold,4)}")
    print("Benign score thresholds:")
    for num_points, score_threshold in zip("-1 -2 -3 -4 -8".split(" "), benign_score_thresholds):
        print(f"{num_points}: {'-------' if np.isnan(score_threshold) else np.round(score_threshold,4)}")
    print(f"estimated prior: {prior:.4f}")

if __name__ == '__main__':
    runExample()
