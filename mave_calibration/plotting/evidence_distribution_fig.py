from pathlib import Path
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fire import Fire
from datetime import datetime

def assign_assay_evidence_strength(score, pathogenic_score_thresholds, benign_score_thresholds):
    if np.isnan(score):
        return 0
    for threshold,points in list(zip(pathogenic_score_thresholds,[1,2,3,4,8]))[::-1]:
        if np.isnan(threshold):
            continue
        if score <= threshold:
            return points
    for threshold,points in list(zip(benign_score_thresholds,[-1,-2,-3,-4,-8]))[::-1]:
        if np.isnan(threshold):
            continue
        if score >= threshold:
            return points
    return 0

def conflicting_interpretations(r):
    """
    Check if a record has conflicting interpretations
    P/LP and B/LB ; P/LP and VUS ; B/LB and VUS ; P/LP and conflicting ; B/LB and conflicting
    If data is mapped at the protein level, this could be a result of different RNA substitutions
    If data is mapped at the RNA level, this is a true conflict

    Parameters
    ----------
    r : pd.Series
        A record from the ClinVar data frame

    Returns
    -------
    bool
        True if there are conflicting interpretations, False otherwise
    """
    return r.num_p_lp > 0 and r.num_b_lb > 0 or \
            r.num_p_lp > 0 and r.num_VUS > 0 or \
            r.num_b_lb > 0 and r.num_VUS > 0 or \
            r.num_p_lp > 0 and r.num_conflicting > 0 or \
            r.num_b_lb > 0 and r.num_conflicting > 0


def is_pathogenic(r):
    return r.num_p_lp > 0 and not conflicting_interpretations(r) and r.clinvar_spliceAI_max <= .5

def is_benign(r):
    return r.num_b_lb > 0 and not conflicting_interpretations(r) and r.clinvar_spliceAI_max <= .5

def is_vus(r):
    return r.num_VUS > 0

def is_conflicting(r):
    return r.num_conflicting > 0

def is_gnomAD(r):
    try:
        return r.AF > 0 and r.spliceAI_scores <= .5
    except AttributeError:
        return r.gnomAD_variants_maxAC_AF > 0 and r.gnomAD_variants_max_spliceAI_score <= .5 

def is_synonymous(r):
    return r.synonymous and r.num_p_lp == 0 and r.clinvar_spliceAI_max <= .5


def generate_evidence_distribution_fig(**kwargs):
    """
    Generate figure plotting the point distribution for a dataset

    Required Arguments
    ---------------------
    - scoreset_filepath : str : path to the scoreset file
    - result_filepath : str : path to the results file

    Optional Arguments
    ---------------------
    - save_dir : str (default None): directory to which to save the figure
    - dataset_name : str : name of the dataset
    - invert_scores : bool (default False) : results are generated on inverted scores, so invert the raw scores before assigning points
    """
    scoreset_filepath = Path(kwargs['scoreset_filepath'])
    result_filepath = Path(kwargs['result_filepath'])
    assert scoreset_filepath.exists()
    assert result_filepath.exists()
    scoreset = pd.read_csv(scoreset_filepath)
    if kwargs.get('invert_scores', False):
        scoreset = scoreset.assign(score = -scoreset.score)
    # read results json containing pathogenic and benign score thresholds
    with open(result_filepath) as f:
        result = json.load(f)
    # assign assay points based on thresholds
    scoreset = scoreset.assign(assay_points=\
        scoreset.score.apply(lambda x: assign_assay_evidence_strength(x,
                                                                        result['pathogenic_score_thresholds'],
                                                                        result['benign_score_thresholds'])))

    # count the number of variants with each evidence strength for each label (P/LP, B/LB, VUS, Conflicting, gnomAD, all) (multi-label)
    count_df = pd.DataFrame.from_records({
    "P/LP" : scoreset.loc[(scoreset.apply(is_pathogenic, axis=1)) & (~scoreset.nonsense),'assay_points'].value_counts(),
    "B/LB" : scoreset.loc[(scoreset.apply(is_benign, axis=1)) & (~scoreset.nonsense),'assay_points'].value_counts(),
    "VUS" : scoreset.loc[(scoreset.apply(is_vus, axis=1)) & (~scoreset.nonsense),'assay_points'].value_counts(),
    "Conflicting" : scoreset.loc[(scoreset.apply(is_conflicting, axis=1)) & (~scoreset.nonsense),'assay_points'].value_counts(),
    "gnomAD" : scoreset.loc[(scoreset.apply(is_gnomAD, axis=1))& (~scoreset.nonsense),'assay_points'].value_counts(),
    "all" : scoreset.loc[~scoreset.nonsense,'assay_points'].value_counts()}).T.fillna(0)
    # add 0 counts for missing points
    for p in [-8,-4,-3,-2,-1,0, 1, 2, 3, 4, 8]:
        if p not in count_df.columns:
            count_df.loc[:,p] = 0
    # sort strength columns
    count_df = count_df.sort_index(axis=1)
    # Generate figure
    cp = sns.color_palette('coolwarm',n_colors=11,)
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    count_df.set_index(count_df.apply(lambda x: f"{x.name} (n={int(x.sum()):,d})",axis=1)).apply(lambda x: 100 * x / x.sum(), axis=1).plot(kind='barh',
                                                                                                                                            stacked=True,
                                                                                                                                            color=cp,
                                                                                                                                            ax=ax)
    ax.legend(title='Strength', bbox_to_anchor=(1.0, .75), loc='upper left')
    ax.set_xlabel("% variants")
    # ax.set_title(f"{dataset_name.replace('_',' ')} Evidence Distribution")

    save_dir = kwargs.get('save_dir', None)
    if save_dir is not None:
        save_dir = Path(save_dir)
        if not save_dir.exists():
            raise ValueError(f"Save directory {save_dir} does not exist")

        # Get current timestamp as a string
        timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        dataset_name = kwargs.get('dataset_name', 'dataset')
        filepath = save_dir / f"{dataset_name}_evidence_distribution_{timestamp_str}.jpg"

        plt.savefig(filepath,format='jpg',bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    Fire(generate_evidence_distribution_fig)