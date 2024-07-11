from mave_calibration.initializations import constrained_gmm_init, gmm_init
from mave_calibration.skew_normal import density_utils
from mave_calibration.evidence_thresholds import get_tavtigian_constant

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mave_calibration.em_opt.utils import get_sample_weights,constrained_em_iteration,em_iteration,get_likelihood
from tqdm.autonotebook import tqdm
import joblib
import json
import os
from pathlib import Path
from fire import Fire


def plot(X, S, current_weights, current_params):
    N_samples = S.shape[1]
    fig, ax = plt.subplots(N_samples, 1, figsize=(10, 10), sharex=True, sharey=True)
    for sample_num in range(N_samples):
        sns.histplot(X[S[:, sample_num]], ax=ax[sample_num], stat="density")
    layer_distributions(X, S, current_weights, current_params, ax, label="estimated")
    return fig, ax


def layer_distributions(X, S, weights_, params_, ax, label="", linestyle="-"):
    cmap = sns.color_palette("tab10", S.shape[1] + 1)
    N_samples = S.shape[1]
    rng = np.arange(X.min(), X.max(), 0.01)
    for sample_num in range(N_samples):
        component_joint_pdfs = density_utils.joint_densities(
            rng, params_, weights_[sample_num]
        )
        for component_num in range(len(params_)):
            ax[sample_num].plot(
                rng,
                component_joint_pdfs[component_num],
                label=f"Component {component_num} {label}",
                color=cmap[component_num],
                linestyle=linestyle,
            )
            mixture_pdf = component_joint_pdfs.sum(axis=0)
        ax[sample_num].plot(
            rng,
            mixture_pdf,
            label=f"Mixture {label}",
            color=cmap[-1],
            linestyle=linestyle,
        )


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
        X = np.concatenate((X, data[sample_name]))
        si = np.zeros((data[sample_name].shape[0], len(sample_names)), dtype=bool)
        si[:, i] = True
        S = np.concatenate((S, si))
    if config["invert_scores"]:
        X = -X
    return X, S


def singleFit(X, S, **kwargs):
    CONSTRAINED = kwargs.get("constrained", True)
    N_components = kwargs.get("n_components", 2)
    N_samples = S.shape[1]
    MAX_N_ITERS = kwargs.get("max_iters", 1000)
    if CONSTRAINED:
        initial_params = constrained_gmm_init(X, n_inits=1, n_components=N_components)
    else:
        initial_params = gmm_init(X, n_inits=1, n_components=N_components)
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
        pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
        pbar.update(1)
        if i > 51 and (np.abs(likelihoods[-50:] - likelihoods[-51:-1]) < 1e-10).all():
            break
    pbar.close()
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


def draw_evidence_figure(X, S, weights, component_params):
    rng = np.arange(X.min(), X.max(), 0.01)
    prior = prior_from_weights(weights)
    f_P = density_utils.joint_densities(rng, component_params, weights[0]).sum(0)
    f_B = density_utils.joint_densities(rng, component_params, weights[1]).sum(0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    ax.plot(rng, f_P / f_B)
    layer_evidence(ax, prior)
    ax.set_yscale("log")
    ax.set_xlabel("Assay Score")
    _ = ax.set_ylabel(r"$\text{LR}^+ $")
    return fig, ax


def save(X, S, best_fit, **kwargs):
    save_path = Path(os.path.join(kwargs.get('save_path'), kwargs.get("dataset_id")))
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
    fig, ax = plot(X, S, weights, component_params)
    fig.savefig(
        os.path.join(save_path, "fit.jpeg"), format="jpeg", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)
    fig, ax = draw_evidence_figure(X, S, weights, component_params)
    fig.savefig(
        os.path.join(save_path, "evidence.jpeg"),
        format="jpeg",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def run(**kwargs):
    X, S = load_data(**kwargs)
    NUM_FITS = kwargs.get("num_fits", 10)
    save_path = kwargs.get("save_path", None)
    best_fit = []
    best_likelihood = -np.inf
    for i in range(NUM_FITS):
        component_params, weights, likelihoods = singleFit(X, S, **kwargs)
        if likelihoods[-1] > best_likelihood:
            best_fit = (component_params, weights, likelihoods)
            best_likelihood = likelihoods[-1]
    if save_path is not None:
        save(X,S,best_fit, **kwargs)


if __name__ == "__main__":
    Fire(run)
