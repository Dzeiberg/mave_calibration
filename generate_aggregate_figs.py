from pathlib import Path
import os
import numpy as np
from main import load_data
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mave_calibration.skew_normal import density_utils
from fire import Fire
from main import prior_from_weights
from mave_calibration.evidence_thresholds import get_tavtigian_constant

data_dir = Path("/mnt/i/bio/mave_curation/")
results_dir = Path("/mnt/d/mave_calibration/results_08_09_24/")
with open(data_dir / "dataset_configs.json") as f:
    dataset_config = json.load(f)
    
data = {sample_name : load_data(dataset_id=sample_name,
                                data_directory=data_dir) \
        for sample_name in os.listdir(data_dir) if os.path.isdir(data_dir / sample_name) and sample_name in dataset_config}


def read_result(r_dir):
    with open(r_dir / "result.json") as f:
        result = json.load(f)
    return result

def get_oob_indices(result, X):
    return np.array(list(set(list(range(X.shape[0]))) - \
                         set(np.concatenate(result["bootstrap_indices"]))))


def get_lrPlus(S, result,dataset_name, pathogenic_sample_num=0):
    controls_idx=data[dataset_name][3]
    f_P = density_utils.joint_densities(S, result['component_params'], result['weights'][pathogenic_sample_num]).sum(0)
    f_B = density_utils.joint_densities(S, result['component_params'], result['weights'][controls_idx]).sum(0)
    return f_P / f_B

def predict_on_oob(result, X,dataset_name):
    oob_indices = get_oob_indices(result, X)
    predictions = np.ones(X.shape[0]) * np.nan
    oob_obs = X[oob_indices]
    predictions[oob_indices] = get_lrPlus(oob_obs, result,dataset_name)
    return predictions

def get_oob_predictions(dataset_name,sample_num=None, return_quantiles=False, return_all=False):
    X,S = data[dataset_name][:2]
    if sample_num is not None:
        x_sample = X[S[:,sample_num]]
    else:
        x_sample = X
    preds = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        if not os.path.isfile(iter_result / "result.json"):
            continue
        result = read_result(iter_result)
        preds.append(predict_on_oob(result, x_sample,dataset_name))
    P = np.stack(preds)
    if return_all:
        return P
    quantiles = np.nanquantile(P, [0.25, .5, 0.75], axis=0)
    if return_quantiles:
        return quantiles[1], quantiles[[0,-1],:]
    return quantiles[1]

def predict(X, dataset_name, posterior=False,return_quantiles=False, return_all=False):
    preds = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        if not os.path.isfile(iter_result / "result.json"):
            continue
        result = read_result(iter_result)
        preds.append(get_lrPlus(X, result,dataset_name))
    P = np.stack(preds)
    if posterior:
        priors = np.array(get_priors(dataset_name)).reshape((-1,1))
        P = P * priors / ((P-1) * priors + 1)
    if return_all:
        return P
    quantiles = np.nanquantile(P, [0.25, .5, 0.75], axis=0)
    if return_quantiles:
        return quantiles[1], quantiles[[0,-1],:]
    return quantiles[1]

def get_priors(dataset_name):
    priors = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        if not os.path.isfile(iter_result / "result.json"):
            continue
        result = read_result(iter_result)
        priors.append(prior_from_weights(np.array(result['weights']),
                                    controls_idx=data[dataset_name][3]))
    return priors

def get_sample_density(X, dataset_name):
    densities = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        result = read_result(iter_result)
        iter_densities = [density_utils.joint_densities(X, result['component_params'], result['weights'][i]).sum(0) \
                          for i in range(len(result['sample_names']))]
        densities.append(iter_densities)
    D = np.stack(densities,axis=1)
    return D

def get_thresholds(dataset_name,posterior=False):
    median_prior = np.quantile(get_priors(dataset_name),.50)
    pathogenic_thresholds_lr,benign_thresholds_lr = thresholds_from_prior(median_prior)
    if posterior:
        # P = P * priors / ((P-1) * priors + 1)
        pathogenic_thresholds = pathogenic_thresholds_lr * median_prior / ((pathogenic_thresholds_lr - 1) * median_prior + 1)
        benign_thresholds = benign_thresholds_lr * median_prior / ((benign_thresholds_lr - 1) * median_prior + 1)
    else:
        pathogenic_thresholds = pathogenic_thresholds_lr
        benign_thresholds = benign_thresholds_lr
    return pathogenic_thresholds, benign_thresholds

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

def assign_strength(score, dataset_name,):
    lrPlus_median, lrPlus_quantiles = predict(score, dataset_name, return_quantiles=True)
    pathogenic_thresholds, benign_thresholds = get_thresholds(dataset_name)
    evidence = []
    for lrP,lrB in zip(lrPlus_quantiles[0], lrPlus_quantiles[1]):
        if lrP > pathogenic_thresholds[-1]:
            evidence.append("PP3_VeryStrong")
        elif lrP > pathogenic_thresholds[-2]:
            evidence.append("PP3_Strong")
        elif lrP > pathogenic_thresholds[-3]:
            evidence.append("PP3_Moderate")
        elif lrP > pathogenic_thresholds[-4]:
            evidence.append("PP3_Supporting")
        elif lrB < benign_thresholds[-1]:
            evidence.append("BS3_VeryStrong")
        elif lrB < benign_thresholds[-2]:
            evidence.append("BS3_Strong")
        elif lrB < benign_thresholds[-3]:
            evidence.append("BS3_Moderate")
        elif lrB < benign_thresholds[-4]:
            evidence.append("BS3_Supporting")
        else:
            evidence.append("Intermediate")
    return evidence
    
def calibration_fig(dataset_name,posterior=False):
    X = data[dataset_name][0]

    xm = X.min()
    xM = X.max()
    rng = np.linspace((xm // .05) * .05-.05, (xM // .05) * .05 + 0.05, 25)
    LR = predict(rng, dataset_name, posterior=posterior,return_all=True)

    fig,ax = plt.subplots(1,1,figsize=(15,5))
    sns.boxplot(data=LR,ax=ax,palette=sns.color_palette("vlag_r",n_colors=LR.shape[1]),whis=1.5,flierprops={"alpha": 0})
    if not posterior:
        ax.set_yscale('log')
    _ = ax.set_xticks(ticks=range(len(rng)),labels=list(map(lambda v: f"{v:.2f}",rng)),rotation=90)
    ax.set_xlabel("Assay Score")
    if not posterior:
        ax.set_ylabel(r"$LR^+$")
    else:
        ax.set_ylabel(r"Posterior Probability Pathogenic")
    pathogenic_thresholds, benign_thresholds = get_thresholds(dataset_name,posterior=posterior)
    for tP,tB,ls in zip(pathogenic_thresholds, benign_thresholds,["-","-.","--",":"]):
        ax.axhline(tP,ls=ls,color='r')
        ax.axhline(tB,ls=ls,color='b')
    if not posterior:
        ax.set_ylim(benign_thresholds[-1] * 1e-2,pathogenic_thresholds[-1] * 1e2)
    else:
        ax.set_ylim(-.01,1.01)
    return fig

def fit_fig(dataset_name):
    N_Samples = data[dataset_name][1].shape[1]
    fig,ax = plt.subplots(N_Samples,1,figsize=(10,3*N_Samples),sharex=True,sharey=True)
    X,S,sample_names,controls_idx = data[dataset_name]
    std=X.std()
    rng = np.linspace(X.min() - std,X.max() + std,1000)
    palette = sns.color_palette("pastel", N_Samples)
    palette_3 = sns.color_palette("dark", N_Samples)
    palette_2 = sns.color_palette("bright", N_Samples)
    D = get_sample_density(rng, dataset_name)
    sample_name_map = dict(p_lp="P/LP", b_lb="B/LB", gnomad="gnomAD", vus="VUS", synonymous="Synonymous",nonsynonymous="Nonsynonymous")
    for i in range(N_Samples):
        sns.histplot(X[S[:,i]],ax=ax[i],stat='density',color=palette[i],label=f"{sample_name_map[sample_names[i]]} (n={S[:,i].sum():,d})")
        ax[i].plot(rng, D[i].mean(0),color=palette_3[i],)
        q = np.nanquantile(D[i], [0.025, .975], axis=0)
        ax[i].fill_between(rng, q[0], q[1], alpha=.5, color=palette_2[i])
        ax[i].legend()
    return fig

def evidence_distr_fig(dataset_name):
    X,S,sample_names,controls_index = data[dataset_name]
    sample_name_map = dict(p_lp="P/LP", b_lb="B/LB", gnomad="gnomAD", vus="VUS", synonymous="Synonymous",nonsynonymous="Nonsynonymous")
    prediction_set = {sample_name_map[k] : predict(X[S[:,i]],dataset_name) for i,k in enumerate(sample_names)}
    obs = load_data(dataset_id=dataset_name,data_directory=data_dir,return_dict=True)
    prediction_set['VUS'] = predict(obs['vus'],dataset_name)

    sns.violinplot(prediction_set,
                orient='h',log_scale=True, bw_adjust=.5, inner='point',palette=sns.color_palette("pastel", len(prediction_set)))

    plt.xlabel(r"$LR^+$")
    pathogenic_thresholds, benign_thresholds = get_thresholds(dataset_name)
    for tP,tB,ls in zip(pathogenic_thresholds, benign_thresholds,["-","-.","--",":"]):
        plt.axvline(tP,ls=ls,color='r',alpha=.5)
        plt.axvline(tB,ls=ls,color='b',alpha=.5)
    plt.gca().invert_xaxis()
    return plt.gcf()

def main(dataset_name,save_dir):
    save_dir = Path(save_dir)
    fig1 = fit_fig(dataset_name)
    savekwargs = dict(format='jpg',dpi=300,bbox_inches='tight')
    fig1.savefig(save_dir / f"{dataset_name}_fit.jpg",**savekwargs)
    plt.close(fig1)
    fig2 = calibration_fig(dataset_name,posterior=True)
    fig2.savefig(save_dir / f"{dataset_name}_calibration.jpg",**savekwargs)
    plt.close(fig2)
    fig3 = evidence_distr_fig(dataset_name)
    fig3.savefig(save_dir / f"{dataset_name}_evidence.jpg",**savekwargs)
    plt.close(fig3)

if __name__ == "__main__":
    Fire(main)