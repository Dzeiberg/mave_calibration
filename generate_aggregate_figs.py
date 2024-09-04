from pathlib import Path
from main import load_data
import json
from fire import Fire
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mave_calibration.skew_normal import density_utils
from main import prior_from_weights
from mave_calibration.evidence_thresholds import get_tavtigian_constant
from tqdm import tqdm
from joblib import Parallel, delayed
from main import run,empirical_iteration,null_iteration
from matplotlib.gridspec import GridSpec
import joblib

def load_results(results_dir,dataset_name,lim=None):
    results = []
    results_dir = Path(results_dir)
    for i,r_dir in tqdm(enumerate(results_dir.glob(f"iter_*/{dataset_name}"))):
        if not (r_dir / "result.json").exists():
            continue
        with open(r_dir / "result.json") as f:
            result = json.load(f)
        results.append(result)
        if lim is not None and i == lim:
            break
    return results

def thresholds_from_prior(prior, point_values=[1,2,3,4,8]):
    exp_vals = 1 / np.array(point_values).astype(float)
    C = get_tavtigian_constant(prior)
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
        exceed = np.where(LR > lr_threshold)[0]
        if len(exceed):
            pathogenic_score_thresholds[strength_idx] = rng[max(exceed)]
    for strength_idx,lr_threshold in enumerate(lr_thresholds_benign):
        exceed = np.where(LR < lr_threshold)[0]
        if len(exceed):
            benign_score_thresholds[strength_idx] = rng[min(exceed)]
    return pathogenic_score_thresholds,benign_score_thresholds

def summarize_thresholds(score_thresholds,q):
    accept = np.isnan(score_thresholds).sum(axis=0) / len(score_thresholds) < .05
    score_thresholds = np.nanquantile(score_thresholds,q=q,axis=0)
    score_thresholds[~accept] = np.nan
    return score_thresholds

def get_sample_density(X, results):
    densities = []
    for result in results:
        iter_densities = [density_utils.joint_densities(X, result['component_params'], result['weights'][i]).sum(0) \
                          for i in range(len(result['sample_names']))]
        densities.append(iter_densities)
    D = np.stack(densities,axis=1)
    return D

def fit_fig(X,S,sample_names,dataset_name,results,ax):
    N_Samples = S.shape[1]
    std=X.std()
    rng = np.arange(X.min() - std,X.max() + std,.01)
    palette = sns.color_palette("pastel", N_Samples)
    palette_3 = sns.color_palette("dark", N_Samples)
    palette_2 = sns.color_palette("bright", N_Samples)
    D = get_sample_density(rng, results)
    sample_name_map = dict(p_lp="P/LP", b_lb="B/LB", gnomad="gnomAD", vus="VUS", synonymous="Synonymous",nonsynonymous="Nonsynonymous")
    bins = np.linspace(X.min(),X.max(),25)
    for i in range(N_Samples):
        sns.histplot(X[S[:,i]],ax=ax[i],stat='density',color=palette[i],bins=bins,label=f"{sample_name_map[sample_names[i]]} (n={S[:,i].sum():,d})")
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


def rejection_test(X,S,sample_names,**kwargs):
    """
    Run the rejection test on the dataset

    Parameters
    ----------
    X : np.ndarray (N,) : The dataset
    S : np.ndarray (N,NSamples) : The sample labels
    sample_names : list (NSamples,) : The sample names

    Required Keyword Arguments
    --------------------------
    dataset_dir : str : The directory of the dataset
    dataset_name : str : The name of the dataset

    Optional Keyword Arguments
    ----------------
    confidence_level : float (default 0.1): The confidence level for the test

    Returns
    -------
    bool : True if the fit is rejected, False otherwise
    """
    CONFIDENCE_LEVEL = kwargs.get('confidence_level',.1)
    qmin,qmax = 100 * CONFIDENCE_LEVEL / 2, 100 * (1 - CONFIDENCE_LEVEL/2)
    test_result = run(data_directory=kwargs['dataset_dir'],
                        dataset_id=kwargs['dataset_name'],
                        bootstrap=False,
                        )
    test_result = dict(zip(['component_params','weights','likehoods'],test_result))
    aucs = [Parallel(n_jobs=-1)(delayed(empirical_iteration)(X[S[:,sample_num]],
                                                                test_result['component_params'],
                                                                test_result['weights'][sample_num]) \
                    for _ in range(1000)) for sample_num in range(S.shape[1])]

    null_aucs = [Parallel(n_jobs=-1)(delayed(null_iteration)(X[S[:,sample_num]]) \
                    for _ in range(1000)) for sample_num in range(S.shape[1])]
    assert len(aucs) == S.shape[1]
    empirical_intervals = [np.percentile(aucs[sample_num], [qmin,qmax]) for sample_num,sampleName in enumerate(sample_names)]
    null_intervals = [np.percentile(null_aucs[sample_num], [qmin,qmax]) for sample_num,sampleName in enumerate(sample_names)]
    for sample_num,(sampleName,empirical, null) in enumerate(zip(sample_names,empirical_intervals,null_intervals)):
        print(f"{sampleName}\n{empirical}\n{null}")
        if empirical[1] < null[0] or empirical[0] > null[1]:
            return True, empirical_intervals, null_intervals       
    return False, empirical_intervals, null_intervals

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
    """
    std = X.std()
    rng = np.arange(X.min() - std,X.max() + std,.01)
    LR_curves = predict(rng,control_sample_index,results,posterior=False,return_all=True)
    priors = np.array(get_priors(results, control_sample_index))
    thresholds_results = Parallel(n_jobs=-1,verbose=10)(delayed(get_score_thresholds)(LR,prior,rng) for LR,prior in list(zip(LR_curves,priors)))
    p_score_thresholds,b_score_thresholds = zip(*thresholds_results)
    p_score_thresholds = np.stack(p_score_thresholds)
    b_score_thresholds = np.stack(b_score_thresholds)
    return p_score_thresholds,b_score_thresholds

def count_violations(X,S,sample_names,final_thresholds_p,final_thresholds_b):
    sample_map = dict(zip(sample_names,range(len(sample_names))))
    p_lp_violations = (X[S[:,sample_map['p_lp']]] > final_thresholds_b[0]).sum()
    b_lb_violations = (X[S[:,sample_map['b_lb']]] < final_thresholds_p[0]).sum()
    return p_lp_violations, b_lb_violations

def main(dataset_name,dataset_dir,results_dir,save_dir,**kwargs):
    """
    Generate the calibration figure for the dataset

    Parameters
    ----------
    dataset_name : str : The name of the dataset
    dataset_dir : str : The directory of the datasets
    results_dir : str : The directory of the results
    save_dir : str : The directory to save the figure

    Optional Keyword Arguments
    --------------------------
    debug : bool : If True, only run on the first 1000 samples
    debug_lim : int : The number of samples to debug on
    max_runs : int (default 10000): The maximum number of runs to use
    """
    # Load Data
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "dataset_configs.json",'r') as f:
        dataset_config = json.load(f)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True,parents=True)
    X,S,sample_names, control_sample_index,mu,sigma = load_data(dataset_id=dataset_name,data_directory=dataset_dir,**dataset_config[dataset_name],return_standardization=True)
    if kwargs.get("debug",False):
        lim = kwargs['debug_lim']
    else:
        lim = kwargs.get('max_runs',10000)
    
    # Load Results
    results = load_results(results_dir,dataset_name,lim=lim)
    # Calculate score thresholds
    if kwargs.get("reload_score_thresholds",True) and (save_dir / f"{dataset_name}_score_thresholds.pkl").exists():
        p_score_thresholds,b_score_thresholds = joblib.load(save_dir / f"{dataset_name}_score_thresholds.pkl")
        p_score_thresholds = np.array(p_score_thresholds)
        b_score_thresholds = np.array(b_score_thresholds)
    else:
        p_score_thresholds,b_score_thresholds = get_score_threshold_mats(X,control_sample_index,results)
        joblib.dump((p_score_thresholds.tolist(),b_score_thresholds.tolist()),save_dir / f"{dataset_name}_score_thresholds.pkl")

    NSamples = S.shape[1]
    fig = plt.figure(layout="constrained", figsize=(8,(NSamples) * 3))

    gs = GridSpec(NSamples + 4, 1, figure=fig,)
    topAxs = [fig.add_subplot(gs[i, 0]) for i in range(NSamples)]

    fit_fig(X,S,sample_names,dataset_name,results,topAxs)
    linestyles = [(0, (1,5)),'dotted','dashed','dashdot','solid']
    final_thresholds_p = summarize_thresholds(p_score_thresholds,.05)
    final_thresholds_b = summarize_thresholds(b_score_thresholds,.95)
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
    # plt.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1, 4))
    ymax = float(max([axi.get_ylim()[1] for axi in topAxs]))
    for axi in topAxs:
        axi.set_ylim(0,ymax)
    p_lp_violations, b_lb_violations = count_violations(X,S,sample_names,final_thresholds_p,final_thresholds_b)
    rejected = False
    if (p_lp_violations / S[:,sample_names.index('p_lp')].sum() > .10) or \
        (b_lb_violations / S[:,sample_names.index('b_lb')].sum() > .10):
        rejected = True
    summary = dict(pathogenic_score_thresholds=final_thresholds_p.tolist(),
                        benign_score_thresholds=final_thresholds_b.tolist(),
                        p_lp_violations=int(p_lp_violations),
                        frac_p_lp_violations=p_lp_violations / S[:,sample_names.index('p_lp')].sum(),
                        b_lb_violations=int(b_lb_violations),
                        frac_b_lb_violations=b_lb_violations / S[:,sample_names.index('b_lb')].sum(),
                        rejected=rejected)
    with open(save_dir / f"{dataset_name}.json",'w') as f:
        json.dump(summary,f)
    savekwargs = dict(format='jpg',dpi=300,bbox_inches='tight')
    suffix = ""
    if kwargs.get("debug",False):
        suffix = "_debug"
    fig.savefig(save_dir / f"{dataset_name}_calibration{suffix}.jpg",**savekwargs)
    plt.close(fig)

if __name__ == "__main__":
    Fire(main)