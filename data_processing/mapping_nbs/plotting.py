import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict,Callable
from pandas import DataFrame

def all_variants(ss):
    return ss.score.values

def p_lp_variants(ss):
    return ss.loc[(ss.num_p_lp > 0) & (ss.clinvar_spliceAI_max < .5),'score'].values

def b_lb_variants(ss):
    return ss.loc[(ss.num_b_lb > 0) & ((ss.clinvar_spliceAI_max < .5)),'score'].values

def gnomad_variants(ss):
    return ss.loc[(ss.gnomAD_variants_maxAC_AF > 0) & (ss.gnomAD_variants_max_spliceAI_score < .5),'score'].values

def vus_variants(ss):
    return ss.loc[ss.num_VUS > 0,'score'].values

def synonymous_variants(ss):
    return ss.loc[ss.synonymous,'score'].values

def plot_distributions(ss,sample_maps : Dict[str,Callable] = dict(all=all_variants,
                                                                    p_lp=p_lp_variants,
                                                                    b_lb=b_lb_variants,
                                                                    gnomAD=gnomad_variants,
                                                                    VUS=vus_variants,
                                                                    synonymous=synonymous_variants)):
    samples = {sample_name : sample_map(ss) for sample_name,sample_map in sample_maps.items()}
    samples = {sample_name : sample for sample_name,sample in samples.items() if len(sample) > 0}
    N_Samples = len(samples)
    X = ss.score.values
    palette = sns.color_palette("pastel", N_Samples)
    bins = np.linspace(X.min(),X.max(),25)
    fig,ax = plt.subplots(N_Samples,1,figsize=(12,3 * N_Samples),sharex=True,sharey=True)
    for i,(axs,(sample_name,sample_observations)) in enumerate(zip(ax,samples.items())):
        label = f"{sample_name} (n={len(sample_observations):,d})"
        sns.histplot(sample_observations,ax=axs,stat='density',color=palette[i],bins=bins,label=label)
        axs.legend()


def plot_samples(samples : Dict[str, DataFrame]):
    X = np.concatenate(list(samples.values()))
    palette = sns.color_palette("pastel", len(samples))
    bins = np.linspace(X.min(),X.max(),25)
    fig,ax = plt.subplots(len(samples),1,figsize=(12,3 * len(samples)),sharex=True,sharey=True)
    for i,(axs,(sample_name,sample_observations)) in enumerate(zip(ax,samples.items())):
        label = f"{sample_name} (n={len(sample_observations):,d})"
        sns.histplot(sample_observations,ax=axs,stat='density',color=palette[i],bins=bins,label=label)
        axs.legend()
    return fig,ax