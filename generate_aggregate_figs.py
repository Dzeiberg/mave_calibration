from pathlib import Path
from mave_calibration.skew_normal import density_utils
import os
import numpy as np
from main import load_data
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fire import Fire
import json

data_dir = Path("/mnt/i/bio/mave_curation/")
results_dir = Path("/mnt/d/mave_calibration/results_08_07_24/")
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


def get_lrPlus(S, result,pathogenic_sample_num=0, benign_sample_num=1):
    f_P = density_utils.joint_densities(S, result['component_params'], result['weights'][pathogenic_sample_num]).sum(0)
    f_B = density_utils.joint_densities(S, result['component_params'], result['weights'][benign_sample_num]).sum(0)
    return f_P / f_B

def predict_on_oob(result, X):
    oob_indices = get_oob_indices(result, X)
    predictions = np.ones(X.shape[0]) * np.nan
    oob_obs = X[oob_indices]
    predictions[oob_indices] = get_lrPlus(oob_obs, result)
    return predictions

def get_oob_predictions(dataset_name,sample_num=None, return_quantiles=False, return_all=False):
    X,S = data[dataset_name][:2]
    if sample_num is not None:
        x_sample = X[S[:,sample_num]]
    else:
        x_sample = X
    preds = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        result = read_result(iter_result)
        preds.append(predict_on_oob(result, x_sample))
    P = np.stack(preds)
    if return_all:
        return P
    quantiles = np.nanquantile(P, [0.25, .5, 0.75], axis=0)
    if return_quantiles:
        return quantiles[1], quantiles[[0,-1],:]
    return quantiles[1]

def predict(X, dataset_name, return_quantiles=False, return_all=False):
    preds = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        if not os.path.isfile(iter_result / "result.json"):
            continue
        result = read_result(iter_result)
        preds.append(get_lrPlus(X, result))
    P = np.stack(preds)
    if return_all:
        return P
    quantiles = np.nanquantile(P, [0.25, .5, 0.75], axis=0)
    if return_quantiles:
        return quantiles[1], quantiles[[0,-1],:]
    return quantiles[1]

def get_sample_density(X, dataset_name):
    densities = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        result = read_result(iter_result)
        iter_densities = [density_utils.joint_densities(X, result['component_params'], result['weights'][i]).sum(0) \
                          for i in range(len(result['sample_names']))]
        densities.append(iter_densities)
    D = np.stack(densities,axis=1)
    return D

def aggregate_thresholds(dataset_name):
    pathogenic_thresholds= []
    benign_thresholds = []
    for iter_result in results_dir.glob(f"iter_*/{dataset_name}"):
        result = read_result(iter_result)
        pathogenic_thresholds.append(result['pathogenic_thresholds'])
        benign_thresholds.append(result['benign_thresholds'])
    return np.nanquantile(np.stack(pathogenic_thresholds), .5,axis=0), np.nanquantile(np.stack(benign_thresholds), .5,axis=0)

def assign_strength(score, dataset_name,):
    lrPlus_median, lrPlus_quantiles = predict(score, dataset_name, return_quantiles=True)
    pathogenic_thresholds, benign_thresholds = aggregate_thresholds(dataset_name)
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
    
def calibration_fig(dataset_name):
    STEP = 0.1
    X = data[dataset_name][0]
    rng = np.arange(np.floor(X.min()),np.ceil(X.max()),STEP)
    # rng = np.arange(X.min(),X.max(),STEP)
    LR = predict(rng, dataset_name, return_all=True)

    fig,ax = plt.subplots(1,1,figsize=(15,5))
    sns.boxplot(data=LR,ax=ax,palette=sns.color_palette("vlag_r",n_colors=LR.shape[1]))

    ax.set_yscale('log')
    _ = ax.set_xticks(ticks=range(len(rng)),labels=list(map(lambda v: f"{v:.2f}",rng)),rotation=90)
    ax.set_xlabel("Assay Score")
    ax.set_ylabel(r"$LR^+$")
    pathogenic_thresholds, benign_thresholds = aggregate_thresholds(dataset_name)
    for tP,tB,ls in zip(pathogenic_thresholds, benign_thresholds,[":","--","-.","-"]):
        ax.axhline(tP,ls=ls,color='r')
        ax.axhline(tB,ls=ls,color='b')
    return fig

def fit_fig(dataset_name):
    N_Samples = data[dataset_name][1].shape[1]
    fig,ax = plt.subplots(N_Samples,1,figsize=(10,3*N_Samples),sharex=True,sharey=True)
    X,S,sample_names = data[dataset_name]
    rng = np.linspace(X.min(),X.max(),1000)
    palette = sns.color_palette("pastel", N_Samples)
    palette_3 = sns.color_palette("dark", N_Samples)
    palette_2 = sns.color_palette("bright", N_Samples)
    D = get_sample_density(rng, dataset_name)
    for i in range(N_Samples):
        sns.histplot(X[S[:,i]],ax=ax[i],stat='density',color=palette[i],label=f"n={S[:,i].sum()}")
        ax[i].set_title(sample_names[i])
        ax[i].plot(rng, D[i].mean(0),color=palette_3[i],)
        q = np.nanquantile(D[i], [0.025, .975], axis=0)
        ax[i].fill_between(rng, q[0], q[1], alpha=.5, color=palette_2[i])
        ax[i].legend()
    return fig

def evidence_distr_fig(dataset_name):
    X, S, sample_names = data[dataset_name]
    N_Samples = S.shape[1]

    sns.violinplot({k : predict(X[S[:,i]],dataset_name) for i,k in enumerate(sample_names)},
               orient='h',log_scale=True, bw_adjust=.5, inner='point',palette=sns.color_palette("pastel", N_Samples))
    # sns.violinplot({k : get_oob_predictions(dataset_name,sample_num=i) for i,k in enumerate(sample_names)},
    #                orient='h',log_scale=True, bw_adjust=.1, inner='point',palette=sns.color_palette("pastel", N_Samples))
    plt.xlabel(r"$LR^+$")
    pathogenic_thresholds, benign_thresholds = aggregate_thresholds(dataset_name)
    for tP,tB,ls in zip(pathogenic_thresholds, benign_thresholds,[":","--","-.","-"]):
        plt.axvline(tP,ls=ls,color='r')
        plt.axvline(tB,ls=ls,color='b')
    plt.gca().invert_xaxis()
    return plt.gcf()

def main(dataset_name,save_dir):
    fig1 = fit_fig(dataset_name)
    savekwargs = dict(format='jpg',dpi=1200,bbox_inches='tight')
    fig1.savefig(save_dir / f"{dataset_name}_fit.jpg",**savekwargs)
    plt.close(fig1)
    fig2 = calibration_fig(dataset_name)
    fig2.savefig(save_dir / f"{dataset_name}_calibration.jpg",**savekwargs)
    plt.close(fig2)
    fig3 = evidence_distr_fig(dataset_name)
    fig3.savefig(save_dir / f"{dataset_name}_evidence.jpg",**savekwargs)
    plt.close(fig3)

if __name__ == "__main__":
    Fire(main)