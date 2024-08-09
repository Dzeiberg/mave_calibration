from mave_calibration.initializations import constrained_gmm_init, gmm_init
from mave_calibration.skew_normal import density_utils
from mave_calibration.evidence_thresholds import get_tavtigian_constant

import matplotlib.pyplot as plt
import scipy.stats as sp
import seaborn as sns
import numpy as np
from mave_calibration.em_opt.utils import get_sample_weights,constrained_em_iteration,em_iteration,get_likelihood
from mave_calibration.em_opt.constraints import density_constraint_violated
from tqdm.autonotebook import tqdm
import joblib
import json
import os
from pathlib import Path
from fire import Fire
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score


def draw_sample(params,weights,N=1):
    samples = []
    for i in range(N):
        k = np.random.binomial(1,weights[1])
        samples.append(sp.skewnorm.rvs(*params[k]))
    return np.array(samples)

def auc_from_samples(x1,x2):
    return roc_auc_score(np.concatenate((np.zeros(x1.shape[0]),np.ones(x2.shape[0]))),np.concatenate((x1,x2)))

def bootstrap_sample(x):
    return np.random.choice(x, size=x.shape[0], replace=True)

def empirical_iteration(sample_observations, params, weights):
    return auc_from_samples(bootstrap_sample(sample_observations),
                          draw_sample(params, weights,
                                      N=len(sample_observations)))

def null_iteration(sample_observations):
    return auc_from_samples(bootstrap_sample(sample_observations),
                          bootstrap_sample(sample_observations))

def plot(X, S, sample_names, current_weights, current_params, **kwargs):
    N_samples = S.shape[1]
    cmap = sns.color_palette("pastel", N_samples)
    fig,ax = plt.subplots(N_samples,1,**kwargs)
    try:
        ax[0]
    except TypeError:
        ax = [ax,]
    sample_name_map = dict(p_lp="Pathogenic/Likely Pathogenic",
                           b_lb="Benign/Likely Benign",
                           vus="VUS",
                           gnomad='gnomAD',
                           synonymous='Synonymous',
                           nonsynonymous='Nonsynonymous',)
    for sample_num in range(N_samples):
        sns.histplot(X[S[:,sample_num]], ax=ax[sample_num],color=cmap[sample_num] ,stat='density',label=f"n={S[:,sample_num].sum():,d}")
        ax[sample_num].spines['top'].set_visible(False)
        ax[sample_num].spines['right'].set_visible(False)
        ax[sample_num].spines['bottom'].set_visible(False)
        ax[sample_num].spines['left'].set_visible(False)
        ax[sample_num].set_ylabel("")
        ax[sample_num].set_yticks(())
        ax[sample_num].legend()
        ax[sample_num].set_title(f"{sample_name_map[sample_names[sample_num]]}".title())
        # ax[sample_num].set_title(f"{sample_names[sample_num].replace('_',' ')} Variants".title())
    layer_distributions(X, S, current_weights, current_params, ax, label='estimated')
    return fig,ax

def layer_distributions(X, S, weights_, params_, ax,label="",linestyle='-'):
    cmap = sns.color_palette("husl",3)
    N_samples = S.shape[1]
    N_components = len(params_)
    rng = np.arange(X.min()-1, X.max()+1, .01)
    for sample_num in range(N_samples):
        component_joint_pdfs = density_utils.joint_densities(rng, params_, weights_[sample_num])
        for component_num in range(N_components):
            ax[sample_num].plot(rng, component_joint_pdfs[component_num],
                label=f"Component {component_num} {label}",
                color=cmap[component_num],
                linestyle=linestyle)
        mixture_pdf = component_joint_pdfs.sum(axis=0)
        ax[sample_num].plot(rng, mixture_pdf, label=f"Mixture {label}",color=cmap[-1],linestyle=linestyle)


def standardize(data,standardize_to):
    mu,sigma = data[standardize_to].mean(),data[standardize_to].std()
    return {k: (v - mu) / sigma for k,v in data.items()}

def load_data(**kwargs):
    """
    Load data from the specified path and return the data and the sample matrix

    Keyword Arguments:
    
    data_directory -- the path to the data directory
    dataset_id -- the id of the dataset, i.e. the directory name within the data directory
    return_dict -- whether to return the data as a dictionary or as a tuple (default False)
    standardize_to -- the sample to standardize to (default None)

    Returns:
    X -- the assay scores (N,)
    S -- the sample matrix (N,S)
    """
    dataset_id = kwargs.get("dataset_id")
    data_directory = kwargs.get("data_directory")
    return_dict = kwargs.get("return_dict", False)
    data = joblib.load(os.path.join(data_directory, dataset_id, "observations.pkl"))
    with open(os.path.join(data_directory, "dataset_configs.json")) as f:
        config = json.load(f)
    standardize_to = config[dataset_id].get("standardize_to", None)
    if standardize_to is None:
        standardize_to = kwargs.get("standardize_to", None)
    if standardize_to is not None:
        print("standardizing to",standardize_to)
        data = standardize(data,standardize_to)
    if return_dict:
        return data
    sample_names = list(config["sample_names"])
    X = np.zeros((0,))
    S = np.zeros((0, len(sample_names)), dtype=bool)
    returned_samples = []
    for i, sample_name in enumerate(sample_names):
        if not len(data[sample_name]):
            continue
        X = np.concatenate((X, data[sample_name]))
        si = np.zeros((data[sample_name].shape[0], len(sample_names)), dtype=bool)
        si[:, i] = True
        S = np.concatenate((S, si))
        returned_samples.append(sample_name)
    if config[dataset_id]["invert"]:
        X = -X
    S = S[:,S.sum(0) > 0]
    return X, S,returned_samples


def singleFit(X, S, **kwargs):
    CONSTRAINED = kwargs.get("constrained", True)
    buffer_stds = kwargs.get('buffer_stds',1)
    obs_std = X.std()
    xlims = (X.min() - obs_std * buffer_stds,
             X.max() + obs_std * buffer_stds)
    N_components = kwargs.get("n_components", 2)
    N_samples = S.shape[1]
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    if CONSTRAINED:
        init_to_sample = kwargs.get("init_to_sample", None)
        if init_to_sample is not None:
            xInit = X[S[:,init_to_sample]]
        else:
            xInit = X
        initial_params = constrained_gmm_init(xInit,**kwargs)
        assert not density_constraint_violated(*initial_params, xlims)
    else:
        initial_params = gmm_init(X,**kwargs)
    W = np.ones((N_samples, N_components)) / N_components
    W = get_sample_weights(X, S, initial_params, W)
    likelihoods = np.array(
        [
            get_likelihood(X, S, initial_params, W) / len(S),
        ]
    )
    try:
        if CONSTRAINED:
            updated_component_params, updated_weights = (
                constrained_em_iteration(X, S, initial_params, W, xlims, iterNum=0)
            )
        else:
            updated_component_params, updated_weights = em_iteration(
                X, S, initial_params, W
            )
    except ZeroDivisionError as e:
        print("ZeroDivisionError")
        return initial_params, W, [*likelihoods, -1 * np.inf]
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(X, S, updated_component_params, updated_weights)
            / len(S),
        ]
    )
    if kwargs.get("verbose",True):
        pbar = tqdm(total=MAX_N_ITERS)
    for i in range(MAX_N_ITERS):
        try:
            if CONSTRAINED:
                updated_component_params, updated_weights = (
                    constrained_em_iteration(
                        X, S, updated_component_params, updated_weights, xlims, iterNum=i+1,
                    )
                )
            else:
                updated_component_params, updated_weights = em_iteration(
                    X, S, updated_component_params, updated_weights
                )
        except ZeroDivisionError as e:
            print("ZeroDivisionError")
            return initial_params, W, [*likelihoods, -1 * np.inf]
        likelihoods = np.array(
            [
                *likelihoods,
                get_likelihood(
                    X, S, updated_component_params, updated_weights
                )
                / len(S),
            ]
        )
        if kwargs.get("verbose",True):
            pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
            pbar.update(1)
        if i > 51 and (np.abs(likelihoods[-50:] - likelihoods[-51:-1]) < 1e-10).all():
            break
    if kwargs.get("verbose",True):
        pbar.close()
    if CONSTRAINED:
        assert not density_constraint_violated(
            updated_component_params[0], updated_component_params[1], xlims
        )
    return updated_component_params, updated_weights, likelihoods


def prior_from_weights(W):
    prior = ((W[2, 0] - W[1, 0]) / (W[0, 0] - W[1, 0])).item()
    return np.clip(prior, 1e-10, 1 - 1e-10)


def P2LR(p, alpha):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return p / (1 - p) * (1 - alpha) / alpha
    # return np.log(p) - np.log(1 - p) + np.log(1-alpha) - np.log(alpha)


def LR2P(lr, alpha):
    return 1 / (1 + np.exp(-1 * (np.log(lr) + np.log(alpha) - np.log(1 - alpha))))


def layer_evidence(ax, prior):
    axlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    xdim = ax.get_xlim()[0] + axlen * 0.05
    C = get_tavtigian_constant(prior)
    pathogenic_evidence_thresholds = np.ones(4) * np.nan
    benign_evidence_thresholds = np.ones(4) * np.nan
    for strength_idx, (i, ls, strength) in enumerate(zip(
        1 / (2 ** np.arange(4)), ["-", "--", "-.", ":"], ["VSt", "St", "Mo", "Su"]
    )):
        pathogenic_evidence_thresholds[strength_idx] = C ** i
        benign_evidence_thresholds[strength_idx] = C ** -i
        ax.axhline(C**i, color="r", linestyle=ls)
        t = ax.text(xdim, C**i, f"P {strength}", fontsize=8)
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
        t2 = ax.text(xdim, C**-i, f"B {strength}", fontsize=8)
        t2.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
        ax.axhline(C ** (-1 * i), color="b", linestyle=ls)
    return C, pathogenic_evidence_thresholds[::-1], benign_evidence_thresholds[::-1]


def draw_evidence_figure(X, S, weights, component_params):
    rng = np.arange(X.min(), X.max(), 0.01)
    prior = prior_from_weights(weights)
    f_P = density_utils.joint_densities(rng, component_params, weights[0]).sum(0)
    f_B = density_utils.joint_densities(rng, component_params, weights[1]).sum(0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    ax.plot(rng, f_P / f_B)
    C, pathogenic_thresholds,benign_thresholds = layer_evidence(ax, prior)
    ax.set_yscale("log")
    ax.set_xlabel("Assay Score")
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,C * 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    _ = ax.set_ylabel(r"$\text{LR}^+ $")
    return fig, ax, pathogenic_thresholds,benign_thresholds


def generate_figures(X, S, sample_names, weights, component_params, **kwargs):
    save_path = os.path.join(kwargs.get("save_path"),
                            kwargs.get("dataset_id"))
    fig, ax = plot(X, S, sample_names, weights, component_params,figsize=(8,6 * len(sample_names)))
    fig.savefig(
        os.path.join(save_path, "fit.jpeg"), 
        format="jpeg",
        dpi=1200,
        transparent=True,
        bbox_inches="tight"
    )
    plt.close(fig)
    fig, ax, pathogenic_thresholds,benign_thresholds = draw_evidence_figure(X, S, weights, component_params)
    fig.savefig(
        os.path.join(save_path, "evidence.jpeg"),
        format="jpeg",
        dpi=1200,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(fig)
    fit_stat_table = get_fit_statistics(X, S, sample_names, weights, component_params)
    with open(os.path.join(save_path, "fit_statistics.json"), "w") as f:
        json.dump(fit_stat_table, f)

    aucs = [Parallel(n_jobs=-1)(delayed(empirical_iteration)(X[S[:,sample_num]], component_params, weights[sample_num]) for _ in range(1000)) for sample_num in range(S.shape[1])]

    null_aucs = [Parallel(n_jobs=-1)(delayed(null_iteration)(X[S[:,sample_num]]) for _ in range(1000)) for sample_num in range(S.shape[1])]

    with open(os.path.join(save_path, "aucs.json"), "w") as f:
        json.dump(
            {
                "empirical": aucs,
                "null": null_aucs,
            },
            f,
        )
    auc_figure(aucs, null_aucs, sample_names, save_path)
    return pathogenic_thresholds,benign_thresholds

def auc_figure(aucs, null_aucs, sample_names, save_path):
    CONFIDENCE_LEVEL = .1
    qmin,qmax = 100 * CONFIDENCE_LEVEL / 2, 100 * (1 - CONFIDENCE_LEVEL/2)
    fig,ax = plt.subplots(2,len(aucs),figsize=(5 * len(aucs),5),sharex='col')
    for i in range(len(aucs)):
        sns.histplot(aucs[i],ax=ax[0,i])
        a = np.array(aucs[i])
        ax[0,i].set_title(f"{sample_names[i]} Empirical AUCs")
        CI = np.percentile(a,[qmin,qmax])
        ax[0,i].hlines(50,CI[0],CI[1],color='r',linestyle='--',label=f"{100 * (1 - CONFIDENCE_LEVEL)}% CI")
        ax[0,i].legend()
        sns.histplot(null_aucs[i],ax=ax[1,i])
        a = np.array(null_aucs[i])
        CI = np.percentile(a,[qmin,qmax])
        ax[1,i].hlines(50,CI[0],CI[1],color='r',linestyle='--',label=f"{100 * (1 - CONFIDENCE_LEVEL)}% CI")
        ax[1,i].legend()
        ax[1,i].set_title(f"{sample_names[i]} Null Distribution")
    fig.savefig(os.path.join(save_path, "auc_rejection_test.jpeg"),format='jpeg',dpi=1200,transparent=True,bbox_inches='tight')

def get_fit_statistics(X, S, sample_names, weights, component_params):
    statistics = {}
    for sample_idx, sample_name in enumerate(sample_names):
        sample_observations = X[S[:,sample_idx]]
        synthetic_observations = draw_sample(component_params,weights[sample_idx],N=len(sample_observations))
        ks_test = sp.kstest(sample_observations, synthetic_observations)
        ks_stat, ks_p = ks_test.statistic, ks_test.pvalue
        xU = np.concatenate((sample_observations, synthetic_observations))
        yU = np.concatenate((np.zeros_like(sample_observations), np.ones_like(synthetic_observations)))
        u_stat, u_p = sp.mannwhitneyu(sample_observations, synthetic_observations)
        auc = roc_auc_score(yU, xU)
        statistics[sample_name] = dict(ks_stat=ks_stat, ks_p=ks_p, u_stat=u_stat, u_p=u_p, auc=auc)
    return statistics

def save(X, S, sample_names, best_fit, bootstrap_indices,**kwargs):
    save_path = Path(os.path.join(kwargs.get('save_path'),
                                    kwargs.get("dataset_id")))
    save_path.mkdir(exist_ok=True, parents=True)
    component_params, weights, likelihoods = best_fit
    pathogenic_thresholds,benign_thresholds = generate_figures(X, S, sample_names, weights, component_params, **kwargs)
    with open(os.path.join(save_path, "result.json"), "w") as f:
        json.dump(
            {
                "component_params": component_params,
                "weights": weights.tolist(),
                "likelihoods": likelihoods.tolist(),
                "config": kwargs,
                "sample_names": sample_names,
                "bootstrap_indices": bootstrap_indices,
                "pathogenic_thresholds": pathogenic_thresholds.tolist(),
                "benign_thresholds": benign_thresholds.tolist(),
            },
            f,
        )
    
    
def reload_result(**kwargs):
    loadpath = os.path.join(kwargs.get("save_path"), kwargs.get("dataset_id"))
    with open(os.path.join(loadpath, "result.json")) as f:
        result = json.load(f)
    return result['component_params'], np.array(result['weights'])

def bootstrap(X, S, **kwargs):
    XBootstrap, SBootstrap = X.copy(), S.copy()
    bootstrap_indices = []
    offset = 0
    for sample in range(S.shape[1]):
        sample_bootstrap_indices = np.random.choice(np.where(S[:,sample])[0], size=S[:,sample].sum(), replace=True)
        bootstrap_indices.append(tuple(list(map(lambda a: a.item(), sample_bootstrap_indices))))
        XBootstrap[offset:offset + len(sample_bootstrap_indices)] = X[sample_bootstrap_indices]
        SBootstrap[offset:offset + len(sample_bootstrap_indices), sample] = True
        offset += len(sample_bootstrap_indices)
    return XBootstrap, SBootstrap, bootstrap_indices

def do_fit(X, S, **kwargs):
    return singleFit(X, S, **kwargs)

def run(**kwargs):
    X, S, sample_names = load_data(**kwargs)
    bootstrap_indices = [tuple(range(si)) for si in S.sum(0)]
    if kwargs.get("bootstrap",True):
        X,S,bootstrap_indices = bootstrap(X,S,**kwargs)
    plot_only = kwargs.get("plot_only", False)
    NUM_FITS = kwargs.get("num_fits", 25)
    save_path = kwargs.get("save_path", None)
    if plot_only:
        component_params, weights = reload_result(**kwargs)
        generate_figures(X, S, sample_names, weights, component_params,**kwargs)
        return
    best_fit = []
    best_likelihood = -np.inf
    fit_results = Parallel(n_jobs=-1)(delayed(do_fit)(X, S, **kwargs) for i in range(NUM_FITS))
    for component_params, weights, likelihoods in fit_results:
        if likelihoods[-1] > best_likelihood:
            best_fit = (component_params, weights, likelihoods)
            best_likelihood = likelihoods[-1]
    if np.isinf(best_likelihood):
        print("No fits succeeded")
        return
    if save_path is not None:
        save(X,S,sample_names, best_fit, bootstrap_indices,**kwargs)
    return best_fit

if __name__ == "__main__":
    Fire(run)
