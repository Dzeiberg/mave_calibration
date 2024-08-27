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

def load_results(results_dir,dataset_name,lim=None):
    results = []
    results_dir = Path(results_dir)
    for i,r_dir in tqdm(enumerate(results_dir.glob(f"iter_*/{dataset_name}"))):
        with open(r_dir / "result.json") as f:
            result = json.load(f)
        results.append(result)
        if lim is not None and i == lim:
            break
    return results

def main(dataset_name,dataset_dir,results_dir,save_dir,**kwargs):
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "dataset_configs.json",'r') as f:
        dataset_config = json.load(f)

    save_dir = Path(save_dir)
    X,S,sample_names, control_sample_index = load_data(dataset_id=dataset_name,data_directory=dataset_dir,**dataset_config[dataset_name])
    if kwargs.get("debug",False):
        lim = 1000
    else:
        lim = None
    results = load_results(results_dir,dataset_name,lim=lim)

    NSamples = S.shape[1]
    fig,ax = plt.subplots(NSamples+1,1,figsize=(10, 3 * (NSamples + 1)),sharex=True,sharey=False)
    fit_fig(X,S,sample_names,dataset_name,results,ax[:-1])
    LR,rng = calibration_fig(X, S, sample_names, control_sample_index,dataset_name,results,ax[-1],posterior=True,lims=ax[0].get_xlim())

    ax[-1].set_ylim(-.01,1.01)
    savekwargs = dict(format='jpg',dpi=300,bbox_inches='tight')
    fig.savefig(save_dir / f"{dataset_name}_calibration.jpg",**savekwargs)
    plt.close(fig)
    # debug here

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
    # fig,ax = plt.subplots(N_Samples,1,figsize=(10,3*N_Samples),sharex=True,sharey=True)
    std=X.std()
    rng = np.linspace(X.min() - std,X.max() + std,1000)
    palette = sns.color_palette("pastel", N_Samples)
    palette_3 = sns.color_palette("dark", N_Samples)
    palette_2 = sns.color_palette("bright", N_Samples)
    D = get_sample_density(rng, results)
    sample_name_map = dict(p_lp="P/LP", b_lb="B/LB", gnomad="gnomAD", vus="VUS", synonymous="Synonymous",nonsynonymous="Nonsynonymous")
    for i in range(N_Samples):
        sns.histplot(X[S[:,i]],ax=ax[i],stat='density',color=palette[i],label=f"{sample_name_map[sample_names[i]]} (n={S[:,i].sum():,d})")
        # ax[i].hist(X[S[:,i]],bins=50,density=True,color=palette[i],label=f"{sample_name_map[sample_names[i]]} (n={S[:,i].sum():,d})")
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

def thresholds_from_prior(prior):
    C = get_tavtigian_constant(prior)
    pathogenic_evidence_thresholds = np.ones(4) * np.nan
    benign_evidence_thresholds = np.ones(4) * np.nan
    for strength_idx, (i, ls, strength) in enumerate(zip(
        1 / (2 ** np.arange(4)), ["-", "--", "-.", ":"], ["VSt", "St", "Mo", "Su"]
    )):
        pathogenic_evidence_thresholds[strength_idx] = C ** i
        benign_evidence_thresholds[strength_idx] = C ** -i
    return pathogenic_evidence_thresholds, benign_evidence_thresholds

def get_thresholds(priors,posterior=False):
    median_prior = np.quantile(priors,0.5)
    pathogenic_thresholds_lr,benign_thresholds_lr = thresholds_from_prior(median_prior)
    if posterior:
        # P = P * priors / ((P-1) * priors + 1)
        pathogenic_thresholds = pathogenic_thresholds_lr * median_prior / ((pathogenic_thresholds_lr - 1) * median_prior + 1)
        benign_thresholds = benign_thresholds_lr * median_prior / ((benign_thresholds_lr - 1) * median_prior + 1)
    else:
        pathogenic_thresholds = pathogenic_thresholds_lr
        benign_thresholds = benign_thresholds_lr
    return pathogenic_thresholds, benign_thresholds

def calibration_fig(X, S, sample_names, control_sample_index,dataset_name,results,ax,posterior=False,lims=None):
    if lims:
        xm,xM = lims
    else:
        xm = X.min() 
        xM = X.max()
    rng = np.arange((xm // .05) * .05-.05,
                    (xM // .05) * .05 + 0.05,
                    .05)
    # rng = np.linspace((xm // .05) * .05-.05, (xM // .05) * .05 + 0.05, 25)
    if posterior:
        priors = np.array(get_priors(results, control_sample_index)).reshape((-1,1))
    LR = predict(rng,control_sample_index,results,posterior=posterior,priors=priors,return_all=True)
    LR_quantiles = np.nanquantile(LR,[0.05,.25,.5,.75,0.95],axis=0)
    ax.plot(rng,LR_quantiles[2],color='black',label=f"prior={np.nanquantile(priors,.5):.3f}")
    for c in [0,1,3,4]:
        ax.plot(rng,LR_quantiles[c],color='black',ls="--")
    # sns.boxplot(data=LR,ax=ax,palette=sns.color_palette("vlag_r",n_colors=LR.shape[1]),whis=1.5,flierprops={"alpha": 0},native_scale=False)
    if not posterior:
        ax.set_yscale('log')
    ax.set_xlabel("Assay Score")
    if not posterior:
        ax.set_ylabel(r"$LR^+$")
    else:
        ax.set_ylabel(r"Posterior Probability Pathogenic")
    pathogenic_thresholds, benign_thresholds = get_thresholds(priors,posterior=posterior)
    for tP,tB,ls in zip(pathogenic_thresholds, benign_thresholds,["-","-.","--",":"]):
        ax.hlines(tP,rng[0],rng[-1],ls=ls,color='r')
        ax.hlines(tB,rng[0],rng[-1],ls=ls,color='b')
    if not posterior:
        ax.set_ylim(benign_thresholds[-1] * 1e-2,pathogenic_thresholds[-1] * 1e2)
    else:
        ax.set_ylim(-.01,1.01)
    ax.legend()
    return LR,rng



if __name__ == "__main__":
    # Fire(main)
    main(dataset_name="vanLoggerenberg_HMBS_erythroid",
        dataset_dir="/mnt/i/bio/mave_curation/",
        results_dir="/mnt/d/mave_calibration/results_08_23_24/",
        save_dir="/mnt/d/mave_calibration/results_08_23_24/figs",
        debug=True)