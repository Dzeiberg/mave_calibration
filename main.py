from mave_calibration.initializations import constrained_gmm_init, gmm_init
from mave_calibration.skew_normal import density_utils
from mave_calibration.evidence_thresholds import get_tavtigian_constant

import matplotlib.pyplot as plt
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

def plot(X, S, sample_names, current_weights, current_params):
    cmap = sns.color_palette("pastel", 3)
    N_samples = S.shape[1]
    fig,ax = plt.subplots(N_samples,1,figsize=(10,10),sharex=True,sharey=True)
    for sample_num in range(N_samples):
        sns.histplot(X[S[:,sample_num]], ax=ax[sample_num],color=cmap[sample_num] ,stat='density')
        ax[sample_num].spines['top'].set_visible(False)
        ax[sample_num].spines['right'].set_visible(False)
        ax[sample_num].spines['bottom'].set_visible(False)
        ax[sample_num].spines['left'].set_visible(False)
        ax[sample_num].set_ylabel("")
        ax[sample_num].set_yticks(())
        ax[sample_num].set_title(f"{sample_names[sample_num].replace('_',' ')} Variants".title())
    layer_distributions(X, S, current_weights, current_params, ax, label='estimated')
    return fig,ax

def layer_distributions(X, S, weights_, params_, ax,label="",linestyle='-'):
    cmap = sns.color_palette("husl",3)
    N_samples = S.shape[1]
    N_components = len(params_)
    rng = np.arange(X.min(), X.max(), .01)
    for sample_num in range(N_samples):
        component_joint_pdfs = density_utils.joint_densities(rng, params_, weights_[sample_num])
        for component_num in range(N_components):
            ax[sample_num].plot(rng, component_joint_pdfs[component_num],
                label=f"Component {component_num} {label}",
                color=cmap[component_num],
                linestyle=linestyle)
        mixture_pdf = component_joint_pdfs.sum(axis=0)
        ax[sample_num].plot(rng, mixture_pdf, label=f"Mixture {label}",color=cmap[-1],linestyle=linestyle)

def load_data(**kwargs):
    """
    Load data from the specified path and return the data and the sample matrix

    Keyword Arguments:
    
    data_directory -- the path to the data directory
    dataset_id -- the id of the dataset, i.e. the directory name within the data directory
    config_name -- the name of the configuration file to load; e.g. missense_config, gnomad_config, etc.

    Returns:
    X -- the assay scores (N,)
    S -- the sample matrix (N,S)
    """
    dataset_id = kwargs.get("dataset_id")
    config_name = kwargs.get("config_name")
    data_directory = kwargs.get("data_directory")
    data = joblib.load(os.path.join(data_directory, dataset_id, "observations.pkl"))
    with open(os.path.join(data_directory, f"{dataset_id}/{config_name}.json")) as f:
        config = json.load(f)

    sample_names = list(config["sample_definitions"].keys())
    X = np.zeros((0,))
    S = np.zeros((0, len(sample_names)), dtype=bool)
    for i, sample_name in enumerate(sample_names):
        if not len(data[sample_name]):
            continue
        X = np.concatenate((X, data[sample_name]))
        si = np.zeros((data[sample_name].shape[0], len(sample_names)), dtype=bool)
        si[:, i] = True
        S = np.concatenate((S, si))
    if config["invert_scores"]:
        X = -X
    return X, S,sample_names


def singleFit(X, S, **kwargs):
    CONSTRAINED = kwargs.get("constrained", True)
    N_components = kwargs.get("n_components", 2)
    N_samples = S.shape[1]
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    if CONSTRAINED:
        initial_params = constrained_gmm_init(X,**kwargs)
        assert not density_constraint_violated(*initial_params, (X.min(), X.max()))
    else:
        initial_params = gmm_init(X,**kwargs)
    W = np.ones((N_samples, N_components)) / N_components
    W = get_sample_weights(X, S, initial_params, W)
    likelihoods = np.array(
        [
            get_likelihood(X, S, initial_params, W) / len(S),
        ]
    )
    xlims = (X.min(), X.max())
    if CONSTRAINED:
        updated_component_params, updated_weights = (
            constrained_em_iteration(X, S, initial_params, W, xlims)
        )
    else:
        updated_component_params, updated_weights = em_iteration(
            X, S, initial_params, W
        )
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(X, S, updated_component_params, updated_weights)
            / len(S),
        ]
    )
    if kwargs.get("verbsose",True):
        pbar = tqdm(total=MAX_N_ITERS)
    for i in range(MAX_N_ITERS):
        if CONSTRAINED:
            updated_component_params, updated_weights = (
                constrained_em_iteration(
                    X, S, updated_component_params, updated_weights, xlims
                )
            )
        else:
            updated_component_params, updated_weights = em_iteration(
                X, S, updated_component_params, updated_weights
            )
        likelihoods = np.array(
            [
                *likelihoods,
                get_likelihood(
                    X, S, updated_component_params, updated_weights
                )
                / len(S),
            ]
        )
        if kwargs.get("verbsose",True):
            pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
            pbar.update(1)
        if i > 51 and (np.abs(likelihoods[-50:] - likelihoods[-51:-1]) < 1e-10).all():
            break
    if kwargs.get("verbsose",True):
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
    for i, ls, strength in zip(
        1 / (2 ** np.arange(4)), ["-", "--", "-.", ":"], ["VSt", "St", "Mo", "Su"]
    ):
        ax.axhline(C**i, color="r", linestyle=ls)
        t = ax.text(xdim, C**i, f"P {strength}", fontsize=8)
        t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
        t2 = ax.text(xdim, C**-i, f"B {strength}", fontsize=8)
        t2.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))
        ax.axhline(C ** (-1 * i), color="b", linestyle=ls)
    return C


def draw_evidence_figure(X, S, weights, component_params):
    rng = np.arange(X.min(), X.max(), 0.01)
    prior = prior_from_weights(weights)
    f_P = density_utils.joint_densities(rng, component_params, weights[0]).sum(0)
    f_B = density_utils.joint_densities(rng, component_params, weights[1]).sum(0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    ax.plot(rng, f_P / f_B)
    C = layer_evidence(ax, prior)
    ax.set_yscale("log")
    ax.set_xlabel("Assay Score")
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(ymin,C * 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    _ = ax.set_ylabel(r"$\text{LR}^+ $")
    return fig, ax


def generate_figures(X, S, sample_names, weights, component_params, **kwargs):
    save_path = os.path.join(kwargs.get("save_path"),
                            kwargs.get("config_name"),
                            kwargs.get("dataset_id"))
    fig, ax = plot(X, S, sample_names, weights, component_params)
    fig.savefig(
        os.path.join(save_path, "fit.jpeg"), 
        format="jpeg",
        dpi=1200,
        transparent=True,
        bbox_inches="tight"
    )
    plt.close(fig)
    fig, ax = draw_evidence_figure(X, S, weights, component_params)
    fig.savefig(
        os.path.join(save_path, "evidence.jpeg"),
        format="jpeg",
        dpi=1200,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(fig)
    fig, ax = fit_quality_figure(X, S, sample_names, weights, component_params)
    fig.savefig(
        os.path.join(save_path, "fit_quality.jpeg"),
        format="jpeg",
        dpi=1200,
        transparent=True,
        bbox_inches="tight",
    )

def fit_quality_figure(X,S, sample_names, weights, component_params, **kwargs):
    individual_fits = {}
    for i, sample_name in enumerate(sample_names):
        xs = X[S[:,i]]
        for _ in range(100):
            try:
                p = singleFit(xs, np.ones((len(xs),1),dtype=bool),max_iters=10000,verbose=False)
            except Exception as e:
                print(e)
                continue
            break
        try:
            individual_fits[sample_name] = p
        except UnboundLocalError:
            try:
                individual_fits[sample_name] = singleFit(xs, np.ones((len(xs),1),dtype=bool),max_iters=10000, n_components=1,constrained=False,verbsose=False)
            except Exception as e:
                print(e)
                continue
    fig, ax = plt.subplots(2,3,figsize=(15,7))
    rng = np.arange(X.min() - .25,X.max() + .25 ,.01)
    fig,ax = plt.subplots(2,3,figsize=(15,7),sharey='row',sharex=True)
    for i, k in enumerate(sample_names):
        sns.histplot(X[S[:,i]],ax=ax[0,i],stat='density',alpha=.5)
        try:
            value = individual_fits[k]
        except KeyError:
            continue
        densities = density_utils.joint_densities(rng, value[0],value[1][0])
        multisample_densities = density_utils.joint_densities(rng, component_params, weights[i])
        P = densities.sum(0)
        Q = multisample_densities.sum(0)
        ax[0,i].plot(rng, Q,color='green', label='multi-sample fit')
        ax[0,i].plot(rng, P,color='orange',label='single-sample fit')
        ax[0,i].legend()
        ax[0,i].set_title(k)
        kl = P * np.log(P/Q)
        ax[1,i].plot(rng, kl,label=f"KL divergence {np.nansum(kl):.3f}")
        ax[1,i].legend()
        ax[1,i].set_xlabel("Assay Score")
    ax[1,0].set_ylabel("Relative Entropy")
    return fig,ax

def save(X, S, sample_names, best_fit, **kwargs):
    save_path = Path(os.path.join(kwargs.get('save_path'),
                                    kwargs.get("config_name"),
                                    kwargs.get("dataset_id")))
    save_path.mkdir(exist_ok=True, parents=True)
    component_params, weights, likelihoods = best_fit
    with open(os.path.join(save_path, "result.json"), "w") as f:
        json.dump(
            {
                "component_params": component_params,
                "weights": weights.tolist(),
                "likelihoods": likelihoods.tolist(),
                "config": kwargs,
            },
            f,
        )
    generate_figures(X, S, sample_names, weights, component_params, **kwargs)
    
def reload_result(**kwargs):
    loadpath = os.path.join(kwargs.get("save_path"), kwargs.get('config_name'), kwargs.get("dataset_id"))
    with open(os.path.join(loadpath, "result.json")) as f:
        result = json.load(f)
    return result['component_params'], np.array(result['weights'])

def bootstrap(X, S, **kwargs):
    XBootstrap, SBootstrap = X.copy(), S.copy()
    offset = 0
    for sample in range(S.shape[1]):
        sample_bootstrap_indices = np.random.choice(np.where(S[:,sample])[0], size=S[:,sample].sum(), replace=True)
        XBootstrap[offset:offset + len(sample_bootstrap_indices)] = X[sample_bootstrap_indices]
        SBootstrap[offset:offset + len(sample_bootstrap_indices), sample] = True
        offset += len(sample_bootstrap_indices)
    return XBootstrap, SBootstrap

def do_fit(X, S, **kwargs):
    return singleFit(X, S, **kwargs)

def run(**kwargs):
    X, S, sample_names = load_data(**kwargs)
    if kwargs.get("bootstrap",True):
        X,S = bootstrap(X,S,**kwargs)
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
    # for i in range(NUM_FITS):
    #     component_params, weights, likelihoods = singleFit(X, S, **kwargs)
    #     if likelihoods[-1] > best_likelihood:
    #         best_fit = (component_params, weights, likelihoods)
    #         best_likelihood = likelihoods[-1]
    if np.isinf(best_likelihood):
        print("No fits succeeded")
        return
    if save_path is not None:
        save(X,S,sample_names, best_fit, **kwargs)
    return best_fit

if __name__ == "__main__":
    Fire(run)
