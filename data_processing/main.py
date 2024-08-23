from pathlib import Path
import pandas as pd
import requests
from splice_ai import querySpliceAI
from clinvar import getClinvar
from gnomad import queryGnomAD
from mavehgvs import Variant
import json
import pickle
from fire import Fire

def process_dataset(dataset_dir, config_file, **kwargs):
    """
    Process a dataset to generate observations and hgvs_pro files

    Required args:
    - dataset_dir: str : Path to the directory containing the dataset
    - config_file: str : Path to the config file

    Optional kwargs:
    - save_results: bool : Whether to save the results to the dataset directory : default True

    """
    with open(config_file) as f:
        config = json.load(f)
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.exists(), "dataset_dir does not exist"
    scoreset_file = dataset_dir / "scoreset.csv"
    assert scoreset_file.exists(), "scoreset.csv does not exist"
    dataset_metadata_file = dataset_dir / "metadata.json"
    assert dataset_metadata_file.exists(), "metadata.json does not exist"
    with open(dataset_metadata_file) as f:
        dataset_metadata = json.load(f)
    uniprot_acc = dataset_metadata['uniprot_acc']
    gene_info = get_gene_info(uniprot_acc)
    scoreset = read_scoreset(scoreset_file)
    scoreset = remove_nonsense(scoreset)
    region_splice_ai_scores = querySpliceAI(gene_info['CHROM'],gene_info['START'],gene_info['STOP'],**config['splice_ai'],write_dir=dataset_dir)
    
    clinvar = getClinvar(**config['clinvar'])
    clinvar = clinvar[clinvar.transcript == gene_info['MANE_RefSeq_nuc']]
    clinvar,clinvarSplice = filter_splice_variants(clinvar,region_splice_ai_scores)
    transcript = gene_info['MANE_RefSeq_nuc']
    clinvarSplice.to_csv(str(dataset_dir / f'clinvar_{transcript}_spliceAI_filtered.csv'),index=False)
    gnomad = queryGnomAD(gene_info['CHROM'],gene_info['START'],gene_info['STOP'],gene_info['HGNC_ID'],**config['gnomad'],write_dir=dataset_dir,
                         **config['external_tools'])
    gnomad,gnomadSplice = filter_splice_variants(gnomad,region_splice_ai_scores)
    gnomadSplice.to_csv(str(dataset_dir / f'gnomad_{transcript}_spliceAI_filtered.csv'),index=False)
    
    observations,hgvs_pro = segment_scoreset(clinvar,gnomad,scoreset)
    if kwargs.get("save_results",True):
        with open(str(dataset_dir / 'observations.pkl'),'wb') as f:
            pickle.dump(observations,f)
        with open(str(dataset_dir / 'hgvs_pro.pkl'),'wb') as f:
            pickle.dump(hgvs_pro,f)
    return observations,hgvs_pro

def read_scoreset(scoreset_file):
    scoreset = pd.read_csv(scoreset_file)
    assert 'score' in scoreset.columns, "score column not found in scoreset"
    assert 'hgvs_pro' in scoreset.columns, "hgvs_pro column not found in scoreset"
    is_synonymous = scoreset.hgvs_pro.apply(lambda x: Variant(x).is_synonymous() or x[2:5] == x[-3:])
    scoreset = scoreset.assign(synonymous=is_synonymous)
    return scoreset

def remove_nonsense(scoreset):
    nonsense = scoreset.hgvs_pro.apply(lambda s: s[-3:] == "Ter")
    return scoreset[~nonsense]

def segment_scoreset(clinvar,gnomad,scoreset):
    p_lp = set(clinvar.loc[clinvar.is_pathogenic,'hgvs_pro'].values)
    b_lb = set(clinvar.loc[clinvar.is_benign,'hgvs_pro'].values)
    vus = set(clinvar.loc[clinvar.ClinicalSignificance == "Uncertain significance",'hgvs_pro'].values)
    mask_p_lp = scoreset.hgvs_pro.isin(p_lp)
    mask_b_lb = scoreset.hgvs_pro.isin(b_lb)
    mask_vus = scoreset.hgvs_pro.isin(vus)
    g = set(gnomad.hgvs_pro.values)
    mask_gnomad = scoreset.hgvs_pro.isin(g)
    # hgvs_pro_missense = set(clinvar.loc[clinvar.snv_category == "missense",'hgvs_pro'].values)
    # mask_missense = scoreset.hgvs_pro.isin(hgvs_pro_missense)
    # TODO: Verify I should remove silent P/LP variants from the synonymous category?
    silent_plp_hgvspro = set(clinvar.loc[(clinvar.snv_category == "silent") & (clinvar.is_pathogenic),'hgvs_pro'].values)
    synonymous_mask = (scoreset.synonymous) & (~scoreset.hgvs_pro.isin(silent_plp_hgvspro))
    observations = dict()
    hgvs_pro = dict()
    for mask,category in zip([mask_p_lp,mask_b_lb,mask_vus,mask_gnomad,synonymous_mask, ~synonymous_mask],
                                ['p_lp','b_lb','vus','gnomad','synonymous','missense']):
        observations[category] = scoreset.loc[mask,'score']
        hgvs_pro[category] = scoreset.loc[mask,'hgvs_pro']
    return observations,hgvs_pro

def filter_splice_variants(df,splice_ai_scores,cutoff=.5):
    df = pd.merge(df,splice_ai_scores,on=['CHROM','POS','REF','ALT'],how='left')
    df.loc[df.spliceAI_score.isna(),'spliceAI_score'] = 0
    nonsplice_mask = df.spliceAI_score < cutoff
    return df[nonsplice_mask], df[~nonsplice_mask]

def get_gene_info(uniprot_acc):
    url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_acc}?"
    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    json = response.json()
    seq = json['sequence']['sequence']
    mane_record = [rec for rec in json['dbReferences'] if rec['type'] == "MANE-Select"][0]
    hgnc_id = [rec for rec in json['dbReferences'] if rec['type'] == "HGNC"][0]['id']
    refseq_nuc = mane_record['properties']['RefSeq nucleotide sequence ID']
    refseq_prot = mane_record['properties']['RefSeq protein sequence ID']
    gene_name = json['gene'][0]['name']['value']

    mane_gtf = pd.read_csv("https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.3.ensembl_genomic.gtf.gz",
                                sep="\t",compression='gzip',header=None)
    mane_record = mane_gtf[(mane_gtf[2] == "gene") & \
        (mane_gtf[8].str.contains(f'gene_name "{gene_name}"',regex=False))].iloc[0]
    CHROM = mane_record[0].replace("chr","")
    START,STOP = mane_record[3],mane_record[4]
    STRAND = mane_record[6]

    return dict(seq=seq,MANE_RefSeq_nuc=refseq_nuc,gene_name=gene_name,MANE_RefSeq_prot=refseq_prot,CHROM=CHROM,START=START,STOP=STOP,HGNC_ID=hgnc_id,STRAND=STRAND)

if __name__ == "__main__":
    Fire(process_dataset)
#     o,v = process_dataset("/mnt/i/bio/mave_curation/findlay_BRCA1_SGE/","data_processing/config.json")