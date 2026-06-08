#!/usr/bin/env python2
'''
#Convert a gtf into a bed file with 12 columns. In the output bed file, each row represents a transcript and blocks in each transcript are exons.
#Input: a gtf file
#Output: a bed file and a tab-separated file indicates the transcript_id and its corresponding gene_name
Usage: python gtf2bed12.py --gtf <gtf> --out <dir>
Code by: Xudong Zou
Start time: 2021-06-09
'''
import argparse
import re

# Function --------------------------------

# input a gtf file and generate a dictionary{transcript:[(s,e),(s,e),...]}
def extract_exon(gtf_file,skip_rows):
	keep_chr = []
	keep_geneType = ['protein_coding','lincRNA','lncRNA']
	for i in range(22):
		keep_chr.append('chr'+str(i+1))
	keep_chr = keep_chr + ['chrX','chrY']
	with open(gtf_file,'r') as fh:
		transcript_exon = {}
		geneid,transcript_id,genename,genetype = '','','',''
		for line in fh.readlines()[skip_rows:]:
			line = line.strip()
			w = line.split("\t")
			chrn = w[0]
			strand = w[6]
			entry_type = w[2]	
			# build match patterns
			geneid_patt_1 = re.search(r'gene_id\s"(\w+)";',w[8])
			geneid_patt_2 = re.search(r'gene_id\s"(\w+\.\d+)";',w[8])# with version number
			transcript_patt_1 = re.search(r'transcript_id\s"(\w+)";',w[8])
			transcript_patt_2 = re.search(r'transcript_id\s"(\w+\.\d+)";',w[8])#with version number
			genename_patt = re.search(r'gene_name\s"(.*?)";',w[8])
			genetype_patt = re.search(r'gene_type\s"(.*?)";',w[8])
			if geneid_patt_1:
				gene_id = geneid_patt_1.group(1)
			elif geneid_patt_2:
				gene_id = geneid_patt_2.group(1)
			else:
				gene_id = "NA"
			if genename_patt:
				genename = genename_patt.group(1)
			else:
				genename = "NA"
			if genetype_patt:
				genetype = genetype_patt.group(1)
			else:
				genetype = "NA"
			if transcript_patt_1 or transcript_patt_2:
				if genetype in keep_geneType:
					transcript_id = transcript_patt_1.group(1) if transcript_patt_1 else transcript_patt_2.group(1)
					transcript_id = transcript_id+":"+gene_id+":"+genename+":"+chrn+":"+strand
					if entry_type == "exon" and chrn in keep_chr:
						exon = (int(w[3]),int(w[4])) #1-based coordinate
						if transcript_id not in transcript_exon:
							transcript_exon[transcript_id] = [exon]
						else:
							transcript_exon[transcript_id].append(exon)
					else:
						continue
				else:
					continue
	return transcript_exon


def extract_Codon(gtf_file,skip_rows,codon_type):
	keep_chr = []
	keep_geneType = ['protein_coding','lincRNA','lncRNA']
	for i in range(22):
		keep_chr.append('chr'+str(i+1))
	keep_chr = keep_chr + ['chrX','chrY']
	with open(gtf_file,'r') as fh:
		transcript_Codon = {}
		geneid,transcript_id,genename,genetype = '','','',''
		for line in fh.readlines()[skip_rows:]:
			line = line.strip()
			w = line.split("\t")
			chrn = w[0]
			strand = w[6]
			entry_type = w[2]	
			# build match patterns
			geneid_patt_1 = re.search(r'gene_id\s"(\w+)";',w[8])
			geneid_patt_2 = re.search(r'gene_id\s"(\w+\.\d+)";',w[8])# with version number
			transcript_patt_1 = re.search(r'transcript_id\s"(\w+)";',w[8])
			transcript_patt_2 = re.search(r'transcript_id\s"(\w+\.\d+)";',w[8])#with version number
			genename_patt = re.search(r'gene_name\s"(.*?)";',w[8])
			genetype_patt = re.search(r'gene_type\s"(.*?)";',w[8])
			if geneid_patt_1:
				gene_id = geneid_patt_1.group(1)
			elif geneid_patt_2:
				gene_id = geneid_patt_2.group(1)
			else:
				gene_id = "NA"
			if genename_patt:
				genename = genename_patt.group(1)
			else:
				genename = "NA"
			if genetype_patt:
				genetype = genetype_patt.group(1)
			else:
				genetype = "NA"
			if transcript_patt_1 or transcript_patt_2:
				if genetype in keep_geneType:
					transcript_id = transcript_patt_1.group(1) if transcript_patt_1 else transcript_patt_2.group(1)
					transcript_id = transcript_id+":"+gene_id+":"+genename+":"+chrn+":"+strand
					if entry_type == codon_type and chrn in keep_chr:
						transcript_Codon[transcript_id] = w[3]
					else:
						continue
				else:
					continue
	return transcript_Codon


def extract_transcript(gtf_file,skip_rows):
	keep_chr = []
	keep_geneType = ['protein_coding','lincRNA','lncRNA']
	for i in range(22):
		keep_chr.append('chr'+str(i+1))
	keep_chr = keep_chr + ['chrX','chrY']
	with open(gtf_file,'r') as fh:
		transcript= {}
		geneid,transcript_id,genename,genetype = '','','',''
		for line in fh.readlines()[skip_rows:]:
			line = line.strip()
			w = line.split("\t")
			chrn = w[0]
			strand = w[6]
			entry_type = w[2]
            # build match patterns
			geneid_patt_1 = re.search(r'gene_id\s"(\w+)";',w[8])
			geneid_patt_2 = re.search(r'gene_id\s"(\w+\.\d+)";',w[8])# with version number
			transcript_patt_1 = re.search(r'transcript_id\s"(\w+)";',w[8])
			transcript_patt_2 = re.search(r'transcript_id\s"(\w+\.\d+)";',w[8])#with version number
			genename_patt = re.search(r'gene_name\s"(.*?)";',w[8])
			genetype_patt = re.search(r'gene_type\s"(.*?)";',w[8])
			if geneid_patt_1:
				gene_id = geneid_patt_1.group(1)
			elif geneid_patt_2:
				gene_id = geneid_patt_2.group(1)
			else:
				gene_id = "NA"
			if genename_patt:
				genename = genename_patt.group(1)
			else:
				genename = "NA"
			if genetype_patt:
				genetype = genetype_patt.group(1)
			else:
				genetype = "NA"
			if transcript_patt_1 or transcript_patt_2:
				if genetype in keep_geneType:
					transcript_id = transcript_patt_1.group(1) if transcript_patt_1 else transcript_patt_2.group(1)
					transcript_id = transcript_id+":"+gene_id+":"+genename+":"+chrn+":"+strand
					if entry_type == "transcript" and chrn in keep_chr:
						t_coord = (int(w[3]),int(w[4])) #1-based coordinate
						transcript[transcript_id] = t_coord
					else:
						continue
				else:
					continue
	return transcript
    
# Main -----------------------------------

parser = argparse.ArgumentParser(description='')
parser.add_argument('-g','--gtf',help="specify a gtf file", required=True)
parser.add_argument('-o','--out_dir',help="specify a directory for the exon output and transcript-genenames",default="gene_annotation.bed")


args = parser.parse_args()

exon_dict = extract_exon(args.gtf,5) # extract exons for each transcript
startCodon_dict = extract_Codon(args.gtf,5,'start_codon')
stopCodon_dict = extract_Codon(args.gtf,5,'stop_codon')
transcript_coord = extract_transcript(args.gtf,5)


# print out gene annotation in bed12 format; transcript_id+":"+gene_id+":"+genename+":"+chrn+":"+strand
fho = open(args.out_dir + "/gene_annotation.bed",'w')
fho2 = open(args.out_dir + "/transcript_to_geneName.txt",'w')
s_codon,e_codon = '',''
for t in exon_dict:
	tid,gid,gname,chrname,strand = t.split(":")
	print >>fho2,"%s\t%s" % (tid,gname)
	t_start,t_end = transcript_coord[t]
	sorted_exons = sorted(exon_dict[t],key=lambda x:x[0])
	exons_starts = [x[0]-t_start for x in sorted_exons]
	exons_sizes = [y-x+1 for x,y in sorted_exons]
	count = len(exons_starts)
    
	exons_starts_str = ",".join(list(map(str,exons_starts))) + ","
	exons_sizes_str = ",".join(list(map(str,exons_sizes))) + ","
    
	if t in startCodon_dict and t in stopCodon_dict:
		s_codon = startCodon_dict[t]
		e_codon = stopCodon_dict[t]
	else:
		s_codon = str(t_start)
		e_codon = str(t_end)
        
	print >>fho, "%s\t%d\t%d\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%s\t%s" % (chrname,t_start-1,t_end,tid,0,strand,int(s_codon)-1,int(e_codon)-1,0,count,exons_sizes_str,exons_starts_str)
fho.close()
fho2.close()

print "Done!"
