import pandas as pd
from datetime import datetime
import subprocess
from pathlib import Path
from pysam import VariantFile

def queryGnomAD(CHROM,START,STOP,HGNC_ID,**kwargs):
    """
    Query gnomAD for missense variants in a gene

    Steps:
    1) Get the chromosomal coordinates of the gene from the MANE GTF file
    2) Use GATK SelectVariants to extract variants in the gene from the gnomAD exomes and genomes VCF files
        2A) Filter for SNPs
        2B) Exclude filtered variants
    3) Use GATK MergeVcfs to combine the exomes and genomes VCF files
    4) Use GATK VariantsToTable to convert the combined VCF file to a TSV file
    5) Manually parse the VEP annotations in the TSV file
    6) Filter for missense variants

    Required args:
    - CHROM: str : The chromosome for which to query gnomAD
    - START: int : The minimum position in the chromosome for which to query gnomAD
    - STOP: int : The maximum position in the chromosome for which to query gnomAD

    Required kwargs:
    - gnomad_vcf_root: str : Path to the root directory of the gnomAD vcf directory
    - picard_filepath: str : Path to the picard.jar file

    Optional kwargs:
    - write_dir: str : Path to the directory where the output files will be written : default "/tmp"

    Returns:
    - missense_df: pd.DataFrame : A DataFrame containing parsed VEP annotations for matched missense variants in gnomAD exomes and genomes
    """
    gnomad_vcf_root = Path(kwargs.get("gnomad_vcf_root"))
    assert gnomad_vcf_root.exists(), "gnomad_vcf_root does not exist"
    write_dir = Path(kwargs.get("write_dir","/tmp"))
    write_dir.mkdir(exist_ok=True)
    java = Path(kwargs.get("java"))
    picard_filepath = Path(kwargs.get("picard_filepath"))
    assert picard_filepath.exists(), "picard_filepath does not exist"

    gnomAD_exomes_filepath = gnomad_vcf_root / f"exomes/gnomad.exomes.v4.1.sites.chr{CHROM}.vcf.bgz"
    gnomAD_genomes_filepath = gnomad_vcf_root / f"genomes/gnomad.genomes.v4.1.sites.chr{CHROM}.vcf.bgz"
    exomes_output_File = write_dir / f"selectvariants_{str(datetime.now()).replace(' ','_')}.exomes.vcf"
    genomes_output_File = write_dir / f"selectvariants_{str(datetime.now()).replace(' ','_')}.genomes.vcf"
    cmd = f"gatk SelectVariants -V {gnomAD_exomes_filepath} -L chr{CHROM}:{START}-{STOP} --select-type-to-include SNP --exclude-filtered --output {exomes_output_File}"
    subprocess.run(cmd.split(" "))
    cmd = f"gatk SelectVariants -V {gnomAD_genomes_filepath} -L chr{CHROM}:{START}-{STOP} --select-type-to-include SNP --exclude-filtered --output {genomes_output_File}"
    subprocess.run(cmd.split(" "))
    output_File = write_dir / f"combinevariants_{str(datetime.now()).replace(' ','_')}.vcf"
    cmd = f'{java} -jar {picard_filepath} MergeVcfs I={exomes_output_File} I={genomes_output_File} O={output_File}'
    subprocess.run(cmd.split(" "))
    tsvout = str(output_File).replace('.vcf','.tsv')
    variants2table = f"gatk VariantsToTable -V {output_File} -F CHROM -F POS -F ID -F REF -F ALT -F QUAL -F FILTER -ASF AC -ASF AF -ASF vep -O {tsvout}"
    subprocess.run(variants2table.split(" "))
    gnomAD_df = pd.read_csv(tsvout,delimiter='\t')
    vep_columns = get_vep_columns_from_vcf_header(output_File)
    vep_df = parse_vep(gnomAD_df,columns=vep_columns)
    gnomAD_df = pd.merge(gnomAD_df,vep_df,left_index=True,right_on='index',validate='one_to_many')
    missense_df = gnomAD_df[gnomAD_df.Consequence == "missense_variant"]
    missense_df = missense_df.assign(hgvs_pro=missense_df.HGVSp.apply(lambda s: s.split(":")[1]))
    missense_df = missense_df.assign(CHROM=missense_df.CHROM.astype(str),
                                     POS=missense_df.POS.astype(str),
                                     REF=missense_df.REF.astype(str),
                                     ALT=missense_df.ALT.astype(str))
    missense_df.to_csv(write_dir / f"gnomad_matches_CHR{CHROM}_{START}_{STOP}.tsv",sep='\t',index=False)
    gene_missense = missense_df[missense_df.HGNC_ID == HGNC_ID]
    gene_missense.to_csv(write_dir / f"gnomad_matches_{HGNC_ID}_CHR{CHROM}_{START}_{STOP}.tsv",sep='\t',index=False)
    return gene_missense

def get_vep_columns_from_vcf_header(vcf_file):
    vcf = VariantFile(vcf_file)
    return vcf.header.info['vep'].description.split("Format: ")[1].split("|")
    
def parse_vep(df,columns):
    vep_series = df.vep.apply(lambda r: list(map(lambda s: dict(zip(columns,s.split('|'))),r.split(","))))
    vep_df = pd.DataFrame(vep_series,index=df.index).explode('vep')
    vep_df = pd.DataFrame.from_records(vep_df.vep.values,index=vep_df.index).reset_index()
    return vep_df