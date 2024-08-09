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

def process_dataset(dataset_dir, config_file):
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
    clinvar = clinvar[clinvar.RefSeq_nuc == gene_info['MANE_RefSeq_nuc']]
    clinvar = filter_splice_variants(clinvar,region_splice_ai_scores)

    gnomad = queryGnomAD(gene_info['CHROM'],gene_info['START'],gene_info['STOP'],**config['gnomad'],write_dir=dataset_dir,
                         picard_filepath=config['external_tools']['picard_filepath'],
                         java_path=config['external_tools']['java'])
    gnomad = filter_splice_variants(gnomad,region_splice_ai_scores)
    
    
    observations,hgvs_pro = segment_scoreset(clinvar,gnomad,scoreset)
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
    p_lp = set(clinvar.loc[clinvar.p_lp,'hgvs_pro'].values)
    b_lb = set(clinvar.loc[clinvar.b_lb,'hgvs_pro'].values)
    vus = set(clinvar.loc[clinvar.vus,'hgvs_pro'].values)
    mask_p_lp = scoreset.hgvs_pro.isin(p_lp)
    mask_b_lb = scoreset.hgvs_pro.isin(b_lb)
    mask_vus = scoreset.hgvs_pro.isin(vus)
    g = set(gnomad.hgvs_pro.values)
    mask_gnomad = scoreset.hgvs_pro.isin(g)
    synonymous_mask = scoreset.synonymous
    observations = dict()
    hgvs_pro = dict()
    for mask,category in zip([mask_p_lp,mask_b_lb,mask_vus,mask_gnomad,synonymous_mask, ~synonymous_mask],['p_lp','b_lb','vus','gnomad','synonymous','nonsynonymous']):
        observations[category] = scoreset.loc[mask,'score']
        hgvs_pro[category] = scoreset.loc[mask,'hgvs_pro']
    return observations,hgvs_pro

def filter_splice_variants(df,splice_ai_scores,cutoff=.5):
    df = pd.merge(df,splice_ai_scores,on=['CHROM','POS','REF','ALT'],how='left')
    df.loc[df.spliceAI_score.isna(),'spliceAI_score'] = 0
    return df[df.spliceAI_score < cutoff]

def get_gene_info(uniprot_acc):
    url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_acc}?"
    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    json = response.json()
    seq = json['sequence']['sequence']
    mane_record = [rec for rec in json['dbReferences'] if rec['type'] == "MANE-Select"][0]
    refseq_nuc = mane_record['properties']['RefSeq nucleotide sequence ID']
    refseq_prot = mane_record['properties']['RefSeq protein sequence ID']
    gene_name = json['gene'][0]['name']['value']

    mane_gtf = pd.read_csv("https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.3.ensembl_genomic.gtf.gz",
                                sep="\t",compression='gzip',header=None)
    mane_record = mane_gtf[(mane_gtf[2] == "gene") & \
        mane_gtf[8].str.contains(gene_name,case=False)].iloc[0]
    CHROM = mane_record[0].replace("chr","")
    START,STOP = mane_record[3],mane_record[4]

    return dict(seq=seq,MANE_RefSeq_nuc=refseq_nuc,gene_name=gene_name,MANE_RefSeq_prot=refseq_prot,CHROM=CHROM,START=START,STOP=STOP)

if __name__ == "__main__":
    Fire(process_dataset)
#     o,v = process_dataset("/mnt/i/bio/mave_curation/findlay_BRCA1_SGE/","data_processing/config.json")