import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path

def querySpliceAI(chrom, position_min, position_max,**kwargs):
        """
        Query SpliceAI for spliceAI scores in a region of the genome

        Required args:
        - chrom: str : The chromosome for which to query SpliceAI
        - position_min: int : The minimum position in the chromosome for which to query SpliceAI
        - position_max: int : The maximum position in the chromosome for which to query SpliceAI

        Required kwargs:
        - spliceAIFilePath: str : Path to the SpliceAI VCF file

        Optional kwargs:
        - write_dir: str : Path to the directory where the output files will be written : default "/tmp"
        
        """
        write_dir = Path(kwargs.get('write_dir',"/tmp"))
        write_dir.mkdir(exist_ok=True)
        spliceAIFilePath = Path(kwargs.get('spliceAIFilePath'))
        assert spliceAIFilePath.exists(), "spliceAIFilePath does not exist"
        output_filepath = write_dir / f'splice_ai_query_result.{str(datetime.now()).replace(" ","_")}.vcf'
        cmd = f"gatk SelectVariants -V {spliceAIFilePath} -L {chrom}:{max(position_min,1)}-{position_max} --output {output_filepath}"
        subprocess.run(cmd.split(" "))
        result_df = pd.read_csv(output_filepath,comment='#',delimiter='\t',header=None,
                        names='CHROM POS ID REF ALT QUAL FILTER INFO'.split(" "),
                            dtype={k : str for k in 'CHROM POS REF ALT'.split(" ")})
        result_df = result_df.assign(spliceAI_score=result_df.INFO.apply(lambda s: max(list(map(float,
                                                                                            s.split("|")[2:6])))))
        return result_df