from pathlib import Path
import logging
from mave_calibration.main import load_data,prior_from_weights
import json
from fire import Fire
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mave_calibration.skew_normal import density_utils
from mave_calibration.evidence_thresholds import get_tavtigian_constant
from mave_calibration.plotting.evidence_distribution_fig import generate_evidence_distribution_fig
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
from typing import List, Dict, Tuple

def load_results(*result_filepaths : List[str]) -> List[Dict]:
    """
    load the results from the json files

    Parameters
    ----------
    *result_filepaths : list of str
        The filepaths to the json files
    Returns
    """
    results = []
    for i,result_file in tqdm(enumerate(result_filepaths)):
        result_file = Path(result_file)
        if not result_file.exists():
            continue
        with open(result_file) as f:
            result = json.load(f)
        results.append(result)
    return results

def thresholds_from_prior(prior, point_values=[1,2,3,4,8]) -> Tuple[List[float]]:
    """
    Get the evidence thresholds (LR+ values) for each point value given a prior

    Parameters
    ----------
    prior : float
        The prior probability of pathogenicity

    
    """
    exp_vals = 1 / np.array(point_values).astype(float)
    C,num_successes = get_tavtigian_constant(prior,return_success_count=True)
    # max number of successes is 17
    max_successes = 17
    if num_successes < max_successes:
        logging.warning(f"Only ({num_successes})/{max_successes} rules for combining evidence are satisfied by constant {C}, found using prior of ({prior:.4f})")
    pathogenic_evidence_thresholds = np.ones(len(point_values)) * np.nan
    benign_evidence_thresholds = np.ones(len(point_values)) * np.nan
    for strength_idx, exp_val in enumerate(exp_vals):
        pathogenic_evidence_thresholds[strength_idx] = C ** exp_val
        benign_evidence_thresholds[strength_idx] = C ** -exp_val
    return pathogenic_evidence_thresholds[::-1], benign_evidence_thresholds[::-1]

def get_score_thresholds(LR,prior,rng):
    lr_thresholds_pathogenic , lr_thresholds_benign = thresholds_from_prior(prior=prior)
    pathogenic_score_thresholds = np.ones(len(lr_thresholds_pathogenic)) * np.nan
    benign_score_thresholds = np.ones(len(lr_thresholds_benign)) * np.nan
    for strength_idx,lr_threshold in enumerate(lr_thresholds_pathogenic):
        if lr_threshold is np.nan:
            continue
        exceed = np.where(LR > lr_threshold)[0]
        if len(exceed):
            pathogenic_score_thresholds[strength_idx] = rng[max(exceed)]
    for strength_idx,lr_threshold in enumerate(lr_thresholds_benign):
        if lr_threshold is np.nan:
            continue
        exceed = np.where(LR < lr_threshold)[0]
        if len(exceed):
            benign_score_thresholds[strength_idx] = rng[min(exceed)]
    return pathogenic_score_thresholds,benign_score_thresholds

def summarize_thresholds(score_thresholds,q):
    # accept_thresholds = np.ones(score_thresholds.shape[0],dtype=bool) * True
    # for i,score_supporting in enumerate(score_thresholds):
    #     if np.isnan(score_supporting) or \
    #         (direction == 'gt' and ((X > score_supporting).sum() / len(X)) > .10) or \
    #         (direction == 'lt' and ((X < score_supporting).sum() / len(X)) > .10):
    #         accept_thresholds[i] = False
    
    # meets threshold in at least 95% of iterations
    accept_evidence_strength = np.isnan(score_thresholds).sum(axis=0) / len(score_thresholds) < .05
    score_thresholds = np.nanquantile(score_thresholds,q=q,axis=0)
    score_thresholds[~accept_evidence_strength] = np.nan
    return score_thresholds

def get_sample_density(X, results):
    densities = []
    for result in results:
        iter_densities = [density_utils.joint_densities(X, result['component_params'], result['weights'][i]).sum(0) \
                          for i in range(len(result['sample_names']))]
        densities.append(iter_densities)
    D = np.stack(densities,axis=1)
    return D

def fit_fig(X,S,sample_names,results,ax, priors=[]):
    N_Samples = S.shape[1]
    std=X.std()
    rng = np.arange(X.min() - std,X.max() + std,.01)
    palette = sns.color_palette("pastel", N_Samples)
    palette_3 = sns.color_palette("dark", N_Samples)
    palette_2 = sns.color_palette("bright", N_Samples)
    D = get_sample_density(rng, results)
    bins = np.linspace(X.min(),X.max(),25)
    for i in range(N_Samples):
        name = sample_names[i]
        label = f"{name} (n={S[:,i].sum():,d})"
        if len(priors) and (name == "gnomAD"):
            label += f" (median prior={np.quantile(priors,.5):.2f})"
        sns.histplot(X[S[:,i]],ax=ax[i],stat='density',color=palette[i],bins=bins,label=label)
        ax[i].plot(rng, D[i].mean(0),color=palette_3[i],)
        q = np.nanquantile(D[i], [0.025, .975], axis=0)
        ax[i].fill_between(rng, q[0], q[1], alpha=.5, color=palette_2[i])
        ax[i].legend()

def get_lrPlus(X, control_sample_index, result, pathogenic_sample_num=0):
    f_P = density_utils.joint_densities(X, result['component_params'],
                                        result['weights'][pathogenic_sample_num]).sum(0)
    f_B = density_utils.joint_densities(X, result['component_params'],
                                        result['weights'][control_sample_index]).sum(0)
    return f_P / f_B

def get_priors(results, control_sample_index):
    priors = []
    for result in results:
        priors.append(prior_from_weights(np.array(result['weights']),
                                    controls_idx=control_sample_index))
    priors = np.array(priors)
    # fill in nans/infs with median
    priors[np.isnan(priors) | np.isinf(priors)] = np.nanquantile(priors,.5)
    return priors

def predict(X,control_sample_index,results,posterior=False, priors=None,return_quantiles=False, return_all=False):
    lrPreds = []
    for result in results:
        lrPreds.append(get_lrPlus(X,control_sample_index,result))
    P = np.stack(lrPreds)
    if posterior:
        assert priors is not None
        P = P * priors / ((P-1) * priors + 1)
    if return_all:
        return P
    quantiles = np.nanquantile(P, [0.25, .5, 0.75], axis=0)
    if return_quantiles:
        return quantiles[1], quantiles[[0,-1],:]
    # return median
    return quantiles[1]

def get_score_threshold_mats(X,control_sample_index,results):
    """
    Calculate the score thresholds corresponding to each evidence strength for each bootstrap iteration

    Parameters
    ----------
    X : np.ndarray
        The assay scores
    control_sample_index : np.ndarray
        The indices of the control samples (usually 1 for B/LB or 4 for synonymous)
    results : list of dict
        Results from the bootstrap iterations

    Returns
    -------
    p_score_thresholds : np.ndarray (n_bootstrap, 5)
        score thresholds for +1, +2, +3, +4, +8 points for each bootstrap iteration

    b_score_thresholds : np.ndarray (n_bootstrap, 5)
        score thresholds for -1, -2, -3, -4, -8 points for each bootstrap iteration

    priors : np.ndarray (n_bootstrap,)
        The prior probabilities for each bootstrap iteration
    """
    std = X.std()
    rng = np.arange(X.min() - std,X.max() + std,.01)
    LR_curves = predict(rng,control_sample_index,results,posterior=False,return_all=True)
    priors = np.array(get_priors(results, control_sample_index))
    thresholds_results = Parallel(n_jobs=-1,verbose=10)(delayed(get_score_thresholds)(LR,prior,rng) for LR,prior in list(zip(LR_curves,priors)))
    p_score_thresholds,b_score_thresholds = zip(*thresholds_results)
    p_score_thresholds = np.stack(p_score_thresholds)
    b_score_thresholds = np.stack(b_score_thresholds)
    return p_score_thresholds,b_score_thresholds, priors

def count_violations(X,S,sample_names,final_thresholds_p,final_thresholds_b):
    sample_map = dict(zip(sample_names,range(len(sample_names))))
    p_lp_violations = (X[S[:,sample_map['P/LP']]] > final_thresholds_b[0]).sum()
    b_lb_violations = (X[S[:,sample_map['B/LB']]] < final_thresholds_p[0]).sum()
    return p_lp_violations, b_lb_violations

def generate_figs(*args,**kwargs):
    """
    Generate the calibration figure for the dataset

    Required Positional Arguments
    -----------------------------
    *args : list of filepaths to result json files

    Required Keyword Arguments
    ----------
    samples_filepath : str
        The path to the data file (csv) containing columns sample_name, score for each observation
    save_dir : str :
        The directory to save the figure

    Optional Keyword Arguments
    --------------------------
    dataset_name : str :
        The name of the dataset to load information from the config file
    config_file : str :
        The path to the config file containing dataset-specific parameters
    processed_scoreset_filepath : str
        Path to the fully processed scoreset (from mapping_nbs directory)
    """
    # Load Data
    config_filepath = kwargs.get('config_file',None)

    if config_filepath is not None:
        config_filepath = Path(config_filepath)
        dataset_name = kwargs['dataset_name']
        if not config_filepath.exists():
            raise FileNotFoundError(f"Config file {config_filepath} not found")
        with open(config_filepath,'r') as f:
            config = json.load(f)
        controls = config[dataset_name]['controls']
        invert_raw_scores = config[dataset_name].get('invert',False)
    else:
        controls = kwargs['control_sample_name']
        invert_raw_scores = False

    save_dir = Path(kwargs['save_dir'])
    save_dir.mkdir(exist_ok=True,parents=True)
    X,S,sample_names = load_data(data_filepath=kwargs['samples_filepath'],)
    
    # Load Results
    results = load_results(*args)
    dataset_name = kwargs.get('dataset_name','dataset')
    control_sample_index = sample_names.index(controls)
    # Calculate score thresholds
    if kwargs.get("reload_score_thresholds",False) and (save_dir / f"{dataset_name}_score_thresholds.pkl").exists():
        p_score_thresholds,b_score_thresholds,priors = joblib.load(save_dir / f"{dataset_name}_score_thresholds.pkl")
        p_score_thresholds = np.array(p_score_thresholds)
        b_score_thresholds = np.array(b_score_thresholds)
        priors = np.array(priors)
    else:
        p_score_thresholds,b_score_thresholds,priors = get_score_threshold_mats(X,control_sample_index,results)
        joblib.dump((p_score_thresholds.tolist(),b_score_thresholds.tolist(),priors.tolist()),save_dir / f"{dataset_name}_score_thresholds.pkl")

    NSamples = S.shape[1]
    fig, topAxs = plt.subplots(NSamples,1,figsize=(8,(NSamples) * 2),sharex=True,sharey=True)

    fit_fig(X,S,sample_names,results,topAxs, priors)
    linestyles = [(0, (1,5)),'dotted','dashed','dashdot','solid']
    qp = kwargs.get('qp',.05)
    qb = kwargs.get('qb',.95)
    final_thresholds_p = summarize_thresholds(p_score_thresholds,qp)
    final_thresholds_b = summarize_thresholds(b_score_thresholds,qb)
    legend_items = []
    for s,linestyle,label in zip(final_thresholds_p,linestyles,['+1','+2','+3','+4','+8']):
        if np.isnan(s):
            continue
        for i,axi in enumerate(topAxs):
            itm = axi.axvline(s,color='r',linestyle=linestyle,label=label)
            if i == 0:
                legend_items.append(itm)
    
    for s,linestyle,label in zip(final_thresholds_b,linestyles,['-1','-2','-3','-4','-8']):
        if np.isnan(s):
            continue
        for i,axi in enumerate(topAxs):
            itm = axi.axvline(s,color='b',linestyle=linestyle,label=label)
            if i == 0:
                legend_items.append(itm)
    ymax = float(max([axi.get_ylim()[1] for axi in topAxs]))
    for axi in topAxs:
        axi.set_ylim(0,ymax)
    p_lp_violations, b_lb_violations = -1, -1
    rejected = False
    if 'P/LP' in sample_names and 'B/LB' in sample_names:
        p_lp_violations, b_lb_violations = count_violations(X,S,sample_names,final_thresholds_p,final_thresholds_b)

        if (p_lp_violations / S[:,sample_names.index('P/LP')].sum() > .10) or \
            (b_lb_violations / S[:,sample_names.index('B/LB')].sum() > .10):
            rejected = True
    summary = dict(pathogenic_score_thresholds=final_thresholds_p.tolist(),
                        benign_score_thresholds=final_thresholds_b.tolist(),
                        p_lp_violations=int(p_lp_violations),
                        b_lb_violations=int(b_lb_violations),
                        rejected=rejected,
                        min_prior=float(priors.min()),
                        max_prior=float(priors.max()),
                        mean_prior=float(priors.mean()),
                        median_prior=float(np.quantile(priors,.5)))
    with open(save_dir / f"{dataset_name}.json",'w') as f:
        json.dump(summary,f)
    savekwargs = dict(format='jpg',dpi=300,bbox_inches='tight')
    fig.savefig(save_dir / f"{dataset_name}_calibration.jpg",**savekwargs)
    plt.close(fig)

    processed_scoreset_filepath = kwargs.get('processed_scoreset_filepath',None)
    if processed_scoreset_filepath is not None:
        generate_evidence_distribution_fig(scoreset_filepath=processed_scoreset_filepath,
                                            result_filepath=save_dir / f"{dataset_name}.json",
                                            save_dir=save_dir,
                                            dataset_name=dataset_name,
                                            invert_scores=invert_raw_scores)

if __name__ == "__main__":
    Fire(generate_figs)