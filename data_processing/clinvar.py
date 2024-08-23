from pathlib import Path
import pandas as pd
from tqdm import tqdm_pandas,tqdm
tqdm.pandas()
from Bio.PDB.Polypeptide import protein_letters_3to1,protein_letters_3to1_extended
import urllib.request
import re
import swifter

residues = set(list(map(lambda k : k.title(), protein_letters_3to1.keys())))

def clinvar_pathogenicity_status(row,pathogenic_or_benign):
    """
    Only >= 1-star P/LP and B/LB annotations are considered
    """
    if pathogenic_or_benign == "pathogenic":
        return row.ClinicalSignificance in {"Pathogenic","Likely pathogenic", "Pathogenic/Likely pathogenic"} and \
            row.ReviewStatus not in {'no clasification provided', 'no assertion criteria provided','no classification for the single variant'}

    elif pathogenic_or_benign == 'benign':
        return row.ClinicalSignificance in {"Benign", "Likely benign", "Benign/Likely benign"} and \
            row.ReviewStatus not in {'no clasification provided', 'no assertion criteria provided','no classification for the single variant'}
    else:
        raise ValueError()

# def getClinvar(year="2024",month="08",use_cached_processed_file=True,cache_dir="/tmp",debug=False):
#     cache_dir = Path(cache_dir)
#     clinvar_ftp_pth = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_{year}-{month}.txt.gz"
#     local_txt_pth = cache_dir / f"variant_summary_{year}-{month}.txt.gz"
#     pkl_pth = cache_dir / f"variant_summary_{year}-{month}.pkl"
#     processed_pkl = cache_dir / f"variant_summary_{year}-{month}_processed.pkl"
#     if use_cached_processed_file and processed_pkl.exists():
#         clinvar_snvs = pd.read_pickle(processed_pkl)
#     else:
#         if pkl_pth.exists():
#             print("loading variant summary pickle")
#             clinvar_variant_summary = pd.read_pickle(pkl_pth)
#             print("loaded")
#         else:
#             if not local_txt_pth.exists():
#                 print("downloading clinvar")
#                 urllib.request.urlretrieve(clinvar_ftp_pth, local_txt_pth)
#             print("reading txt.gz file")
#             clinvar_variant_summary = pd.read_csv(local_txt_pth,delimiter="\t",compression='gzip')
#             print("saving as pickle")
#             pd.to_pickle(clinvar_variant_summary,pkl_pth)
#         # ONLY ANNOTATED SNV
#         clinvar_snvs = clinvar_variant_summary[clinvar_variant_summary.Type == "single nucleotide variant"]
#         # Only protein variants
#         clinvar_snvs = clinvar_snvs[clinvar_snvs.Name.apply(is_protein_variant)]
#         # Annotate P/LP and B/LB when >= 1-star
#         clinvar_snvs = clinvar_snvs.assign(is_pathogenic=\
#                                            clinvar_snvs.apply(lambda row: clinvar_pathogenicity_status(row,'pathogenic'),axis=1),
#                                            is_benign=clinvar_snvs.apply(lambda row: clinvar_pathogenicity_status(row,'benign'),axis=1))
#         clinvar_snvs = clinvar_snvs.assign(GeneID = clinvar_snvs.GeneID.astype(str))
        
#         clinvar_snvs = clinvar_snvs.assign(HGVSp="p."+clinvar_snvs.Name.str.split("p.",regex=False).str[1].str.slice(0,-1),
#                                         RefSeq_nuc=clinvar_snvs.Name.str.split("(",regex=False).str[0])
#         clinvar_snvs = clinvar_snvs.assign(is_missense=clinvar_snvs.Name.apply(is_missense))
#         clinvar_snvs = clinvar_snvs.assign(is_nonsense=clinvar_snvs.Name.apply(is_nonsense))
#         clinvar_snvs = clinvar_snvs.assign(is_unknown=clinvar_snvs.Name.apply(is_unknown))
#         clinvar_snvs = clinvar_snvs.assign(is_silent=clinvar_snvs.Name.apply(is_silent))
#         clinvar_snvs = clinvar_snvs.assign(is_other_protein_variant=clinvar_snvs.Name.apply(is_other_protein_variant))
#         clinvar_snvs = annotate(clinvar_snvs)
#         parsed_dicts = clinvar_snvs.Name.apply(parse_clinvar_name)
        
#         parsed_names = pd.DataFrame.from_records(parsed_dicts.values,index=clinvar_snvs.index)
#         parsed_names.columns = ["Name_"+col for col in parsed_names.columns]
#         if clinvar_snvs.shape[0] != parsed_names.shape[0]:
#             raise ValueError(f"Mismatch in number of rows between clinvar_snvs and parsed_names: {clinvar_snvs.shape[0]} vs {parsed_names.shape[0]}")
#         clinvar_snvs = pd.concat([clinvar_snvs,parsed_names],axis=1)
#         # Cast types and rename to align with other files
#         clinvar_snvs = clinvar_snvs.assign(
#             CHROM=clinvar_snvs.Chromosome.astype(str),
#             POS=clinvar_snvs.PositionVCF.astype(str),
#             REF=clinvar_snvs.ReferenceAlleleVCF.astype(str),
#             ALT=clinvar_snvs.AlternateAlleleVCF.astype(str),
#             hgvs_pro=clinvar_snvs.HGVSp)
#         pd.to_pickle(clinvar_snvs,processed_pkl)
#         if debug:
#             return clinvar_snvs, parsed_names
#     return clinvar_snvs,

def getClinvar(year="2024",month="08",use_cached_processed_file=True,write_to_cache=True,cache_dir="/tmp"):
    cache_dir = Path(cache_dir)
    clinvar_ftp_pth = f"https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_{year}-{month}.txt.gz"
    local_txt_pth = cache_dir / f"variant_summary_{year}-{month}.txt.gz"
    pkl_pth = cache_dir / f"variant_summary_{year}-{month}.pkl"
    processed_pkl = cache_dir / f"variant_summary_{year}-{month}_processed.pkl"
    if use_cached_processed_file and processed_pkl.exists():
        clinvar_processed = pd.read_pickle(processed_pkl)
    else:
        if pkl_pth.exists():
            print("loading variant summary pickle")
            clinvar_variant_summary = pd.read_pickle(pkl_pth)
            print("loaded")
        else:
            if not local_txt_pth.exists():
                print("downloading clinvar")
                urllib.request.urlretrieve(clinvar_ftp_pth, local_txt_pth)
            print("reading txt.gz file")
            clinvar_variant_summary = pd.read_csv(local_txt_pth,delimiter="\t",compression='gzip')
            print("saving as pickle")
            pd.to_pickle(clinvar_variant_summary,pkl_pth)
        # ONLY ANNOTATED SNV
        snv = clinvar_variant_summary[clinvar_variant_summary["Type"] == "single nucleotide variant"]
        # Categorize SNV type (missense, nonsense, silent, unknown, other_protein_variant, non_protein_variant)
        snv = snv.assign(snv_category = snv.Name.progress_apply(categorize_snv))
        # Annotate P/LP and B/LB when >= 1-star
        snv = snv.assign(
                    is_pathogenic = (snv.ClinicalSignificance.isin({"Pathogenic","Likely pathogenic", "Pathogenic/Likely pathogenic"})) & \
                    (~snv.ReviewStatus.isin({'no clasification provided', 'no assertion criteria provided','no classification for the single variant'})),
                is_benign = (snv.ClinicalSignificance.isin({"Benign", "Likely benign", "Benign/Likely benign"})) & \
                    (~snv.ReviewStatus.isin({'no clasification provided', 'no assertion criteria provided','no classification for the single variant'})))
        # Cast GeneID as string
        snv.GeneID = snv.GeneID.astype(str)
        # Parse Name column to extract transcript, gene symbol, DNA sequence substitution, and protein substitution
        parsed_snv_names = snv.Name.progress_apply(parse_clinvar_name)
        parsed_name_df = pd.DataFrame.from_records(parsed_snv_names.values,index=snv.index)
        clinvar_processed = pd.concat((snv,parsed_name_df),axis=1)
        if (clinvar_processed.Name != clinvar_processed.name).any():
            raise ValueError("Name and name do not match")
        clinvar_processed.drop(columns=["name"],inplace=True)
        clinvar_processed = clinvar_processed.assign(
            CHROM=clinvar_processed.Chromosome.astype(str),
            POS=clinvar_processed.PositionVCF.astype(str),
            REF=clinvar_processed.ReferenceAlleleVCF.astype(str),
            ALT=clinvar_processed.AlternateAlleleVCF.astype(str))
        if write_to_cache:
            clinvar_processed.to_pickle(processed_pkl)
    return clinvar_processed

def annotate(clinvar):
    p_lp = (clinvar.is_pathogenic) & (clinvar.is_missense)
    b_lb = (clinvar.is_benign) & (clinvar.is_missense)
    vus = (clinvar.ClinicalSignificance == "Uncertain significance") & (clinvar.is_missense)
    conflicting = (clinvar.ClinicalSignificance == "Conflicting classifications of pathogenicity") & (clinvar.is_missense)
    clinvar = clinvar.assign(p_lp=p_lp,b_lb=b_lb,vus=vus,conflicting=conflicting)
    return clinvar

def parse_protein_variant(s):
    """
    Parse protein conseequence of a variant

    Parameters
    ----------
    s : str
        The input string to parse
    
    Returns
    -------
    reference_aa : str
    """
    # SPECIAL CASE: no protein
    if s == "p.0":
        raise ValueError("No protein")
    
    # Define a regular expression pattern to match the number in the string
    pattern = re.compile(r'(\D*)(\d+)(\D*)')
    
    # Search for the pattern in the input string
    match = pattern.search(s)
    
    if match:
        # Extract the three parts from the match groups
        before_number = match.group(1)
        number = match.group(2)
        after_number = match.group(3)
        wt_aa = before_number.replace("p.","")
        after_number = after_number.replace(")","")
        if wt_aa not in residues and wt_aa not in set(("Ter","Sec","Pyl")):
            raise ValueError(f"Expecting the value before the number to be a residue, not '{wt_aa}', as found in '{s}'")
        try:
            number = int(number)
        except (ValueError,TypeError):
            raise ValueError(f"Cannot convert {number} to an integer")
        if after_number[0] == "_":
            raise ValueError("Looks like variant causes a new translation initiation site")
        if "ext" in after_number:
            raise ValueError("Looks like an extension")
        if "^" in after_number:
            raise ValueError("Uncertain")
        if "/" in after_number:
            raise ValueError("Mosaic")
        return wt_aa, number, after_number
    else:
        # Return the original string if no number is found
        raise ValueError("Not a supported protein variant")

def is_protein_variant(name):
    return "(p." in name

def is_silent(name):
    if not is_protein_variant(name):
        return False
    protein_variant = name.split("p.")[1]
    try:
        wt_aa, position, variant = parse_protein_variant(protein_variant)
    except ValueError:
        return False
    return variant == "="

def is_nonsense(name):
    if not is_protein_variant(name):
        return False
    protein_variant = name.split("p.")[1]
    try:
        wt_aa, position, variant = parse_protein_variant(protein_variant)
    except ValueError:
        return False
    if variant == "*":
        return True
    if variant == "Ter":
        return True
    return False

def is_unknown(name):
    if not is_protein_variant(name):
        return False
    protein_variant = name.split("p.")[1]
    try:
        wt_aa, position, variant = parse_protein_variant(protein_variant)
    except ValueError:
        return False
    return variant == "?" or variant == "Xaa"

def is_missense(name):
    if not is_protein_variant(name):
        return False
    protein_variant = name.split("p.")[1]
    try:
        wt_aa, position, variant = parse_protein_variant(protein_variant)
    except ValueError:
        return False
    return variant in residues

def is_other_protein_variant(name):
    if not is_protein_variant(name):
        return False
    protein_variant = name.split("p.")[1]
    try:
        wt_aa, position, variant = parse_protein_variant(protein_variant)
    except ValueError:
        return True
    return variant not in {"=", "*", "Ter", "?", "Xaa"} and variant not in residues

def categorize_snv(name):
    if not is_protein_variant(name):
        return "non_protein_variant"
    protein_variant = name.split("p.")[1]
    try:
        wt_aa, position, variant = parse_protein_variant(protein_variant)
    except ValueError:
        return "other_protein_variant"
    if variant in residues:
        return "missense"
    if variant == "=":
        return "silent"
    if variant == "*" or variant == "Ter":
        return "nonsense"
    if variant == "?" or variant == "Xaa":
        return "unknown"
    return "unsupported"

def parse_clinvar_name(record_name):
    """
    Extract transcript, gene symbol (if exists), DNA sequence substitution, and protein substitution (if exists) from a ClinVar record name.
    
    Args:
        record_name (str): The ClinVar record name (e.g., "NM_015697.9(COQ2):c.30G>A (p.Arg10_Lys11=)" or "c.30G>A (p.Arg10_Lys11=)" or "NM_015697.9:c.30G>A" or "NM_015697.9(COQ2):c.30G>A").
    
    Returns:
        dict: A dictionary containing 'transcript', 'gene_symbol', 'dna_substitution', and 'protein_substitution'.
    """
    # Regular expression pattern to match the ClinVar record format with optional transcript, gene symbol, and protein substitution
    pattern = re.compile(
        r'(?:(?P<transcript>NM_\d+\.\d+|\w+_\d+))?(?:\s*(?:\((?P<gene_symbol>[^\)]+)\)|(?P<gene_symbol_no_paren>[^\s:]+)))?:(?P<dna_substitution>c\.[^\s]+)(?:\s*\((?P<protein_substitution>p\.[^\)]+)\))?'
    )
    
    # Search for the pattern in the input string
    match = pattern.search(record_name)
    
    if match:
        # Extract the parts from the match groups
        transcript = match.group('transcript') or ''
        gene_symbol = match.group('gene_symbol') or match.group('gene_symbol_no_paren') or ''
        hgvs_nuc = match.group('dna_substitution')
        hgvs_pro = match.group('protein_substitution') or ''
        
        return {
            'name': record_name,
            'transcript': transcript,
            'geneSymbol': gene_symbol,
            'hgvs_nuc': hgvs_nuc,
            'hgvs_pro': hgvs_pro
        }
    else:
        # Return None or raise an error if the format is incorrect
        return {
            'name': record_name,
            'transcript': '',
            'geneSymbol': '',
            'hgvs_nuc': '',
            'hgvs_pro': ''}