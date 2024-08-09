from pathlib import Path
import pandas as pd
from Bio.PDB.Polypeptide import protein_letters_3to1
import urllib.request

def clinvar_pathogenicity_status(row,pathogenic_or_benign):
    """
    Only >= 1-star P/LP and B/LB annotations are considered
    """
    if pathogenic_or_benign == "pathogenic":
        return row.ClinicalSignificance in ["Pathogenic","Likely pathogenic", "Pathogenic/Likely pathogenic"] and \
            row.ReviewStatus not in {'no clasification provided', 'no assertion criteria provided','no classification for the single variant'}

    elif pathogenic_or_benign == 'benign':
        return row.ClinicalSignificance in ["Benign", "Likely benign", "Benign/Likely benign"] and \
            row.ReviewStatus not in {'no clasification provided', 'no assertion criteria provided','no classification for the single variant'}
    else:
        raise ValueError()

def getClinvar(year="2024",month="07",use_cache=True,cache_dir="/tmp"):
    cache_dir = Path(cache_dir)
    clinvar_ftp_pth = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_{year}-{month}.txt.gz"
    local_txt_pth = cache_dir / f"clinvar_{year}_{month}_snvs.txt.gz"
    pkl_pth = cache_dir / f"clinvar_{year}_{month}_snvs.pkl"
    if use_cache and pkl_pth.exists():
        clinvar_snvs = pd.read_pickle(pkl_pth)
    else:
        if local_txt_pth.exists():
            clinvar_variant_summary = pd.read_csv(local_txt_pth,
                                                  delimiter="\t",compression='gzip')
        else:
            urllib.request.urlretrieve(clinvar_ftp_pth, local_txt_pth)
            clinvar_variant_summary = pd.read_csv(local_txt_pth,delimiter="\t",compression='gzip')
        clinvar_snvs = clinvar_variant_summary[clinvar_variant_summary.Type == "single nucleotide variant"]
        clinvar_snvs = clinvar_snvs[clinvar_snvs.Name.str.contains("p.",regex=False)]
        clinvar_snvs = clinvar_snvs.assign(is_pathogenic=\
                                           clinvar_snvs.apply(lambda row: clinvar_pathogenicity_status(row,'pathogenic'),axis=1),
                                           is_benign=clinvar_snvs.apply(lambda row: clinvar_pathogenicity_status(row,'benign'),axis=1))
        clinvar_snvs = clinvar_snvs.assign(GeneID = clinvar_snvs.GeneID.astype(str))
        
        clinvar_snvs = clinvar_snvs.assign(HGVSp="p."+clinvar_snvs.Name.str.split("p.",regex=False).str[1].str.slice(0,-1),
                                        RefSeq_nuc=clinvar_snvs.Name.str.split("(",regex=False).str[0])
        residues = set(protein_letters_3to1.keys())
        clinvar_snvs = clinvar_snvs.assign(is_missense=(clinvar_snvs.HGVSp.str.slice(2,5).str.upper().apply(lambda v: v in residues)) & (clinvar_snvs.HGVSp.str.slice(-3).str.upper().apply(lambda v: v in residues)))
        clinvar_snvs = annotate(clinvar_snvs)
        # Cast types and rename to align with other files
        clinvar_snvs = clinvar_snvs.assign(
            CHROM=clinvar_snvs.Chromosome.astype(str),
            POS=clinvar_snvs.PositionVCF.astype(str),
            REF=clinvar_snvs.ReferenceAlleleVCF.astype(str),
            ALT=clinvar_snvs.AlternateAlleleVCF.astype(str),
            hgvs_pro=clinvar_snvs.HGVSp)
        pd.to_pickle(clinvar_snvs,pkl_pth)
    return clinvar_snvs

def annotate(clinvar):
    p_lp = (clinvar.is_pathogenic) & (clinvar.is_missense)
    b_lb = (clinvar.is_benign) & (clinvar.is_missense)
    vus = (clinvar.ClinicalSignificance == "Uncertain significance") & (clinvar.is_missense)
    conflicting = (clinvar.ClinicalSignificance == "Conflicting classifications of pathogenicity") & (clinvar.is_missense)
    clinvar = clinvar.assign(p_lp=p_lp,b_lb=b_lb,vus=vus,conflicting=conflicting)
    return clinvar
