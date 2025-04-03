import pandas as pd
import qtl.io
from pathlib import Path
import gzip
import numpy as np
from collections import defaultdict
# Function to convert gtf to bed
def gtf_to_tss_bed(annotation_gtf, feature='gene', exclude_chrs=[], phenotype_id='gene_id'):
    """
    Parse genes from GTF and return DataFrame with min start and max end positions.
    For each gene, uses the smallest position as start and largest as end position.
    """
    
    # Data collections
    gene_data = defaultdict(lambda: {'chr': '', 'start': float('inf'), 'end': 0, 'strand': ''})
    gene_names = {}  # To store gene_name for each gene_id
    
    if annotation_gtf.endswith('.gz'):
        opener = gzip.open(annotation_gtf, 'rt')
    else:
        opener = open(annotation_gtf, 'r')
        
    with opener as gtf:
        for row in gtf:
            row = row.strip().split('\t')
            if row[0][0] == '#' or row[2] != feature: 
                continue  # Skip header or non-matching features
            
            # Parse attributes
            attributes = defaultdict()
            for a in row[8].replace('"', '').split(';')[:-1]:
                kv = a.strip().split(' ')
                if kv[0] != 'tag':
                    attributes[kv[0]] = kv[1]
                else:
                    attributes.setdefault('tags', []).append(kv[1])
            
            # Get gene identifiers
            curr_gene_id = attributes['gene_id']
            curr_gene_name = attributes['gene_name']
            gene_names[curr_gene_id] = curr_gene_name
            
            # Update gene data
            data = gene_data[curr_gene_id]
            if data['chr'] == '':  # First entry for this gene
                data['chr'] = row[0]
                data['strand'] = row[6]
            
            # Update min start and max end
            start_pos = int(row[3]) - 1  # Convert to 0-based for BED
            end_pos = int(row[4])
            data['start'] = min(data['start'], start_pos)
            data['end'] = max(data['end'], end_pos)
    
    # Convert to dataframe format
    chrom, start, end, ids, names = [], [], [], [], []
    for gene_id, data in gene_data.items():
        chrom.append(data['chr'])
        start.append(data['start'])
        end.append(data['end'])
        ids.append(gene_id)
        names.append(gene_names[gene_id])
    
    # Create DataFrame based on phenotype_id
    if phenotype_id == 'gene_id':
        bed_df = pd.DataFrame({
            'chr': chrom,
            'start': start,
            'end': end,
            'gene_id': ids
        }, columns=['chr', 'start', 'end', 'gene_id'], index=ids)
    elif phenotype_id == 'gene_name':
        bed_df = pd.DataFrame({
            'chr': chrom,
            'start': start,
            'end': end,
            'gene_id': names
        }, columns=['chr', 'start', 'end', 'gene_id'], index=names)
    
    # Filter out excluded chromosomes
    mask = np.ones(len(chrom), dtype=bool)
    for k in exclude_chrs:
        mask = mask & (bed_df['chr'] != k)
    bed_df = bed_df[mask]
    
    # Sort by chromosome and start position
    bed_df = bed_df.groupby('chr', sort=False, group_keys=False).apply(lambda x: x.sort_values(['start', 'end']))
    
    return bed_df
    
def prepare_bed(df, bed_template_df, chr_subset=None):
    bed_df = pd.merge(bed_template_df, df, left_index=True, right_index=True)
    bed_df = bed_df.groupby('#chr', sort=False, group_keys=False).apply(lambda x: x.sort_values(['start', 'end']))
    if chr_subset is not None:
        bed_df = bed_df[bed_df.chr.isin(chr_subset)]
    return bed_df

def load_and_preprocess_data(input_path, drop_columns, sep="\t"):
    df = pd.read_csv(input_path, sep=sep, skiprows=0)
    dc = [col for col in df.columns if col in drop_columns] # Take interscet between df.columns and drop_columns
    df = df.drop(dc,axis = 1) # drop the intersect
    if len(df.columns) < 2:
        raise ValueError("There are too few columns in the loaded dataframe, please check the delimiter of the input file. The default delimiter is tab")
    return df

def rename_samples_using_lookup(df, lookup_path):
    sample_participant_lookup = Path(lookup_path)
    if sample_participant_lookup.is_file():
        sample_participant_lookup_s = pd.read_csv(sample_participant_lookup, sep="\t", index_col=1, dtype={0:str,1:str})
        df.rename(columns=sample_participant_lookup_s.to_dict()["genotype_id"], inplace=True)
    return df

def load_bed_template(input_path, phenotype_id_type):
    if sum(gtf_to_tss_bed(input_path, feature='gene',phenotype_id = "gene_id").index.duplicated()) > 0:
        raise valueerror(f"gtf file {input_path} needs to be collapsed into gene model by reference data processing module")

    bed_template_df_id = gtf_to_tss_bed(input_path, feature='transcript', phenotype_id="gene_id")
    bed_template_df_name = gtf_to_tss_bed(input_path, feature='transcript', phenotype_id="gene_name")
    bed_template_df = bed_template_df_id.merge(bed_template_df_name, on=["chr", "start", "end"])
    bed_template_df.columns = ["#chr", "start", "end", "gene_id", "gene_name"]
    bed_template_df = bed_template_df.set_index(phenotype_id_type, drop=False)

    return bed_template_df