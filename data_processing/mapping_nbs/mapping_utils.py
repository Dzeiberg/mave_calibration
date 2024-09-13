import joblib
import pandas as pd
from typing import List

def load_mapping_data(**kwargs):
    """
    Load mapping data from a pickle file

    Required Kwargs
    ----------
    mapping_filepath : str
        Path to the mapping data pickle file
        default : "/data/dzeiberg/mave_calibration/cache/mapping_data.pkl"

    Returns
    -------
    metadata : pd.DataFrame
        Dataset to uniprot_acc dataframe
    gene_info : pd.DataFrame
        uniprot_acc to gene info (transcripts, HGNC, symbols, etc.)
    gnomAD_df : pd.DataFrame
        gnomAD data frame
    spliceAI_df : pd.DataFrame
        spliceAI data frame
    clinvar_df : pd.DataFrame
        clinvar data frame
    """
    mapping_data = joblib.load(kwargs.get('mapping_filepath',"/data/dzeiberg/mave_calibration/cache/mapping_data.pkl"))
    metadata = mapping_data['metadata']
    gene_info = mapping_data['gene_info']
    gnomAD_df = mapping_data['gnomad_df']
    spliceAI_df = mapping_data['spliceAI_df']
    clinvar_df = mapping_data['clinvar_df']
    return metadata, gene_info, gnomAD_df, spliceAI_df, clinvar_df

def summarize_clinvar_group(grp):
    p_lp = grp.is_pathogenic.sum()
    b_lb = grp.is_benign.sum()
    conflicting = (grp.ClinicalSignificance == "Conflicting classifications of pathogenicity").sum()
    VUS = (grp.ClinicalSignificance == "Uncertain significance").sum()
    names = "|".join(grp.Name.values)
    spliceAI_max = grp.spliceAI_score.max()
    return pd.Series(dict(num_p_lp=p_lp,num_b_lb=b_lb,num_conflicting=conflicting,num_VUS=VUS,clinvar_names=names,clinvar_records=len(grp),clinvar_spliceAI_max=spliceAI_max))

def get_clinvar_summaries(clinvar_df,refseq_transcript):
    """
    ClinVar summaries for a given refseq transcript

    Parameters
    ----------
    clinvar_df : pd.DataFrame
        ClinVar data frame
    
    refseq_transcript : str
        RefSeq transcript ID (e.g. NM_XXXXXX)
    
    Returns
    -------
        hgvs_pro_summaries : pd.DataFrame
            Summary of ClinVar data for each protein HGVS variant
            num_p_lp, num_b_lb, num_conflicting, num_VUS, clinvar_names, clinvar_records, clinvar_spliceAI_max
    """
    assert refseq_transcript[:2] == "NM" and ("." not in refseq_transcript), "Refseq transcript must be in the form NM_XXXXXX"
    group_summaries = clinvar_df[(clinvar_df.transcript_base == refseq_transcript)].groupby("hgvs_pro").progress_apply(summarize_clinvar_group)
    hgvs_pro_summaries = pd.DataFrame(group_summaries)
    hgvs_pro_summaries = hgvs_pro_summaries[hgvs_pro_summaries.index.str.len() > 0]
    return hgvs_pro_summaries

def translate_refseq_to_ensembl(refseq_transcript,**kwarg):
    """
    Translate a RefSeq transcript to an Ensembl transcript using Ensembl v.112

    Parameters
    ----------
    refseq_transcript : str
        RefSeq transcript ID (e.g. NM_XXXXXX or NM_XXXXXX.X)
    
    Required kwargs
    ---------------
    transcript_table_filepath : str
        Path to the Ensembl transcript table
        default "/data/dzeiberg/mave_calibration/cache/transcript_mapping_table.tsv"

    Returns
    -------
    ensembl_transcripts : list
        List of Ensembl transcript Stable IDs
    """
    with_version = "." in refseq_transcript
    table = pd.read_csv(kwarg.get("transcript_table_filepath",
        "/data/dzeiberg/mave_calibration/cache/transcript_mapping_table.tsv"),sep="\t")
    if not with_version:
        Ensembl_transcript_stable_ids = table.loc[table.display_label.str.contains(refseq_transcript+".",regex=False),'stable_id'].values
    else:
        Ensembl_transcript_stable_ids = table.loc[table.display_label == refseq_transcript,'stable_id'].values
    return Ensembl_transcript_stable_ids

def gather_gnomAD_info(gnomAD_df : pd.DataFrame,
                        Ensembl_transcript_stable_ids : List[str] = [],
                        RefSeq_transcript_ids : List[str] = [],):
    """
    Gather gnomAD information for a given Ensembl transcript

    Parameters
    ----------
    gnomAD_df : pd.DataFrame
        gnomAD data frame
    
    Ensembl_transcript_stable_id : List[str]
        Ensembl transcript Stable ID

    Returns
    -------
    gnomAD_info : pd.DataFrame
        gnomAD v4.1 genomes + exomes summarized at the amino acid substitution level
        gnomAD_variants_maxAC_AF, gnomAD_variants_max_spliceAI_score, gnomAD_variants_VCF_INFO
         - gnomAD_variants_maxAC_AF : allele frequency of the record with the highest allele count
         - gnomAD_variants_max_spliceAI_score : maximum spliceAI score
         - gnomAD_variants_VCF_INFO : GRCh38 genomic coordinates of the variants (e.g. "1:12345:A:T|2:34567:G:C")
    """
    Ensembl_transcript_stable_ids = [x.split(".")[0] for x in Ensembl_transcript_stable_ids]
    RefSeq_transcript_ids = [x.split(".")[0] for x in RefSeq_transcript_ids]
    gnomAD_variants = gnomAD_df[(gnomAD_df.Feature_base.isin(set(Ensembl_transcript_stable_ids))) | \
                                (gnomAD_df.Feature_base.isin(RefSeq_transcript_ids))].reset_index().groupby('hgvs_pro')
    gnomAD_variants_maxAC_AF = gnomAD_variants.apply(lambda grp: grp.loc[grp.AC.idxmax(),'AF'],include_groups=False)
    gnomAD_variants_max_spliceAI_score = gnomAD_variants.spliceAI_scores.max()
    gnomAD_variants_VCF_INFO = gnomAD_variants.apply(lambda grp: "|".join(grp.loc[:,['CHROM',"POS",'REF','ALT']].astype(str).apply(":".join, axis=1).values))
    return pd.DataFrame(dict(gnomAD_variants_maxAC_AF=gnomAD_variants_maxAC_AF,
                            gnomAD_variants_max_spliceAI_score=gnomAD_variants_max_spliceAI_score,
                            gnomAD_variants_VCF_INFO=gnomAD_variants_VCF_INFO))


