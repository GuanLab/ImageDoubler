
import os
import pandas as pd
from glob import glob


def same_seq(file_r1, file_r2):
    return file_r1.replace("_R1.fastq", "") == file_r2.replace("_R2.fastq", "")


def get_kallisto_cmd(file1, file2, outdir):
    cmd = f"""
        kallisto quant -i ../refs/ensemblGrch38_kallisto_index \
                       -o {outdir} \
                       -t 16 \
                       {file1} {file2}
    """
    return cmd


files_r1 = sorted(glob("split/*/*R1.fastq"))
files_r2 = sorted(glob("split/*/*R2.fastq"))

columns = pd.read_csv("columns.csv")
barcode2cols = dict(zip(columns["index"], columns["column"]))

for file_r1, file_r2 in zip(files_r1, files_r2):
    assert same_seq(file_r1, file_r2), f"They are from different sequence results: {file_r1}, {file_r2}"
    
    # 106891_TAAGGCGA_S1_ROW01_R1.fastq
    filename_elements = os.path.basename(file_r1).split("_")
    the_col = barcode2cols[filename_elements[1]]
    the_row = filename_elements[3]
    outdir = f"gene/{the_col}_{the_row}/"
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    cmd = get_kallisto_cmd(file_r1, file_r2, outdir)
    os.system(cmd)
    
    



