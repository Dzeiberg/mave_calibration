import pandas as pd
from datetime import datetime
import subprocess
from pathlib import Path

def queryGnomAD(CHROM,START,STOP,**kwargs):
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
    java = Path(kwargs.get("java_path"))
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
    variants2table = f"gatk VariantsToTable -V {output_File} -F CHROM -F POS -F ID -F REF -F ALT -F QUAL -F FILTER -ASF AC -ASF AF -ASF vep -O {str(output_File).replace('.vcf','.tsv')}"
    subprocess.run(variants2table.split(" "))
    gnomAD_df = pd.read_csv(str(output_File).replace('.vcf','.tsv'),delimiter='\t')
    vep_df = parse_vep(gnomAD_df)
    gnomAD_df = pd.merge(gnomAD_df,vep_df,left_index=True,right_on='index',validate='one_to_many')
    missense_df = gnomAD_df[gnomAD_df.Consequence == "missense_variant"]
    missense_df = missense_df.assign(hgvs_pro=missense_df.HGVSp.apply(lambda s: s.split(":")[1]))
    missense_df = missense_df.assign(CHROM=missense_df.CHROM.astype(str),
                                     POS=missense_df.POS.astype(str),
                                     REF=missense_df.REF.astype(str),
                                     ALT=missense_df.ALT.astype(str))
    missense_df.to_csv(write_dir / f"gnomad_matches_CHR{CHROM}_{START}_{STOP}.tsv",sep='\t',index=False)
    return missense_df

def parse_vep(df):
    vep_columns = "Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|ALLELE_NUM|DISTANCE|STRAND|FLAGS|VARIANT_CLASS|MINIMISED|SYMBOL_SOURCE|HGNC_ID|CANONICAL|TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|GENE_PHENO|SIFT|PolyPhen|DOMAINS|HGVS_OFFSET|GMAF|AFR_MAF|AMR_MAF|EAS_MAF|EUR_MAF|SAS_MAF|AA_MAF|EA_MAF|ExAC_MAF|ExAC_Adj_MAF|ExAC_AFR_MAF|ExAC_AMR_MAF|ExAC_EAS_MAF|ExAC_FIN_MAF|ExAC_NFE_MAF|ExAC_OTH_MAF|ExAC_SAS_MAF|CLIN_SIG|SOMATIC|PHENO|PUBMED|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|MOTIF_SCORE_CHANGE|LoF|LoF_filter|LoF_flags|LoF_info".split("|")
    vep_series = df.vep.apply(lambda r: list(map(lambda s: dict(zip(vep_columns,s.split('|'))),r.split(","))))
    vep_df = pd.DataFrame(vep_series,index=df.index).explode('vep')
    vep_df = pd.DataFrame.from_records(vep_df.vep.values,index=vep_df.index).reset_index()
    return vep_df