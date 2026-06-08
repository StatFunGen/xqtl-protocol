#based on MRAID, science advances,2022 https://xzlab.org/papers/2022_Yuan_etal_SA.pdf 

# install.packages("devtools")
# library(devtools)
# install_github("yuanzhongshang/MRAID")

# install.packages("LDlinkR")
#install.packages("data.table")

out<-"outputs/01-MRAID_test"
dir.create(out,recursive = T)


library(MRAID)
library(LDlinkR)
library(data.table)
library(stringr)
setDTthreads(threads = 0)
getDTthreads() #28

library(biomaRt)

#functions####

geneMart<-useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")
snpMart<- useEnsembl(biomart = "snps", 
                     dataset = "hsapiens_snp")

TransEnsembltoSymbol<-function(ensembl_ids){
  require(biomaRt)
  require(data.table)
  return(data.table::data.table(biomaRt::getBM(attributes = c('ensembl_gene_id', 'hgnc_symbol'),
                                               filters = 'ensembl_gene_id', 
                                               values = ensembl_ids, 
                                               mart = geneMart)))
}
GetChrInfos<-function(hgnc_symbols){
  require(biomaRt)
  require(data.table)
  return(data.table::data.table(biomaRt::getBM(attributes = c('chromosome_name','start_position','end_position', 'hgnc_symbol'),
                                               filters = 'hgnc_symbol',
                                               values = hgnc_symbols, 
                                               mart = geneMart)))
}
GetRSID<-function(chr_coords){
  getBM(attributes = c('refsnp_id', 'chr_name', 'chrom_start', 'chrom_end', 'allele'),
        filters = c('chromosomal_region'), 
        values = chr_coords, #e.g 1:10020:10020 
        mart = snpMart)
}
#analysis####
#toy dataset####
#Summary data with 100 candidate correlated SNPs, n1=30000, n2=30000, 
data_path<-"xqtl-protocol/code/mendelian_randomization/MRAID/example/" #clone the github repository of MRAID (https://github.com/yuanzhongshang/MRAID)

#load the Zscore vector for the exposure
x<-fread(file.path(data_path,"Zscore_1.txt"))
Zscore_1<-x$x

#load the Zscore vector for the outcome
y<-fread(file.path(data_path,"Zscore_2.txt"))
Zscore_2<-y$x


#load the LD matrix in exposure GWAS data 
zx<-fread(file.path(data_path,"Sigma1sin.txt"))
Sigma1sin<-as.matrix(data.frame(zx,row.names = "V1"))

#load the LD matrix in outcome GWAS data 
zy<-fread(file.path(data_path,"Sigma2sin.txt"))
Sigma2sin<-as.matrix(data.frame(zy,row.names = "V1"))

#load the sample size
samplen1=30000
samplen2=30000

#run MRAID 
?MRAID
result<-MRAID(Zscore_1, Zscore_2, Sigma1sin, Sigma2sin,
              samplen1, samplen2, 
              Gibbsnumber=1000,burninproportion=0.2,
              pi_beta_shape=0.5,
              pi_beta_scale=4.5,pi_c_shape=0.5,pi_c_scale=9.5,pi_1_shape=0.5,pi_1_scale=1.5,pi_0_shape=0.05,pi_0_scale=9.95,option=1)

result
# $causal_effect
# [1] 0.07167195
# 
# $causal_pvalue
# [1] 0.2958619
# 
# $correlated_pleiotropy_effect
# [1] -0.00347539
# 
# $sigmabeta
# [1] 0.01766178
# 
# $sigmaeta
# [1] 0.01002248
# 
# $sigma_error_1
# [1] 0.9914672
# 
# $sigma_error_2
# [1] 1.000612


#test causality for APOE and BIN1, as a positive control####
#APOE####
#SNPs associated with APOE expre (genome wide significance 10^-8)
#in cis-bulk eQTL####
#APOE == ENSG00000130203

res_apoe<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/rna-seq/BU/ROSMAP_DLPFC/eQTL/fine_mapping_susie/tsv_files/demo.ENSG00000130203.unisusie.fit.all_pheno.tsv")

res_chr19<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/rna-seq/BU/ROSMAP_DLPFC/eQTL/association_scan/norminal_qced_files/dlpfc_batch_all.rnaseqc.low_expression_filtered.outlier_removed.tmm.expression.bed.processed_phenotype.per_chrom_dlpfc_batch_all.rnaseqc.ROSMAP_covariates.ROSMAP_NIA_WGS.pca.PEER.txt.19.norminal.cis_long_table.txt")

res_chr19[pvalue<10^(-8)] #101787

res_chr19[pvalue<10^(-8)&molecular_trait_id=="ENSG00000130203"] #0


res_chr19[molecular_trait_id=="ENSG00000130203"][order(pvalue)] #pvalue < 10^-3
res_chr19[molecular_trait_id=="ENSG00000130203"&pvalue<0.001][order(pvalue)] #4 only

#better with pseudobulk ?####
res_pseudo_micro<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/single-cell-rna-seq/pseudo_bulk/eight_celltypes_sumstat/Mic.log2cpm.cov_pca.resid.PEER.cov.emprical.cis_sumstats.txt")

3/0.003649635 #822
4/0.004866180 #822 
#n=822
res_pseudo_micro[molecular_trait_id=="APOE"] 

#lack the SNPs level informations
pseudo_micro<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/single-cell-rna-seq/pseudo_bulk/eight_celltypes_bedgz/Mic.log2cpm.bed.gz")
dim(pseudo_micro) #7834  428
head(pseudo_micro[,1:10])


system("tar xvzf ../../xQTL/ftp/ftp_fgc_xqtl/projects/single-cell-rna-seq/pseudo_bulk/eight_celltypes_sumstat/nominal_sumstat.tar.gz data/pseudobulk_eqtl")
res_pseudo_micro19<-fread("data/pseudobulk_eqtl/nominal_sumstat/Mic.log2cpm.cov_pca.resid.PEER.cov.19.norminal.cis_long_table.txt.gz")

res_pseudo_micro19[pvalue<10^(-8)&molecular_trait_id=="APOE"] #57

res_pseudo_micro19[pvalue<10^(-5)&molecular_trait_id=="APOE"] #109, ok test this one

#stringent (pvalue<10^(-8))
snps_apoe<-res_pseudo_micro19[pvalue<10^(-8)&molecular_trait_id=="APOE"]$variant

#gwas
res_gwas19<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/ADGWAS_Bellenguez_2022/ADGWAS2022.chr19.sumstat.tsv")
res_gwas19[,variant:=str_replace(variant,'_',':')]
res_gwas19[variant%in%snps_apoe] #43/57

com_snps_apoe<-res_gwas19[variant%in%snps_apoe]$variant #43/57

?MRAID

#need z score
res_gwas19[,beta_mean:=mean(beta)]
res_gwas19[,beta_sd:=sd(beta)]
res_gwas19[,beta_z:=(beta-beta_mean)/beta_sd]


res_pseudo_micro19[,beta_mean:=mean(beta,na.rm = T),by=.(molecular_trait_id)]
res_pseudo_micro19[,beta_sd:=sd(beta,na.rm = T),by=.(molecular_trait_id)]
res_pseudo_micro19[,beta_z:=(beta-beta_mean)/beta_sd,by=.(molecular_trait_id)]
res_pseudo_micro19[pvalue<10^(-8)&molecular_trait_id=="APOE"]

#need LD 
#will used LDlink db first

cat(paste(str_extract(com_snps_apoe,'chr[0-9]+:[0-9]+'),collapse = "\n"))
#extract LD matrix from 
#https://ldlink.nci.nih.gov/?tab=ldmatrix

system("wget -O data/ldlink_snp_apoe.txt https://ldlink.nci.nih.gov/tmp/r2_2068.txt")

ld_mat<-fread("data/ldlink_snp_apoe.txt")

n_eqtl<-unique(res_pseudo_micro19[variant%in%com_snps_apoe&molecular_trait_id=="APOE"]$n)
res_gwas19[,n_tot:=n_cases+n_controls]
n_gwas<-round(mean(res_gwas19[variant%in%com_snps_apoe]$n_tot))

result<-MRAID(res_pseudo_micro19[molecular_trait_id=="APOE"][com_snps_apoe,on="variant"]$beta_z,
              res_gwas19[com_snps_apoe,on="variant"]$beta_z,
              Sigma1sin=as.matrix(data.frame(ld_mat,row.names = "RS_number")),
              Sigma2sin=as.matrix(data.frame(ld_mat,row.names = "RS_number")),
              samplen1=n_eqtl, #411
              samplen2=n_gwas, #486115
              Gibbsnumber=1000,burninproportion=0.2,
              pi_beta_shape=0.5,
              pi_beta_scale=4.5,
              pi_c_shape=0.5,pi_c_scale=9.5,pi_1_shape=0.5,pi_1_scale=1.5,pi_0_shape=0.05,pi_0_scale=9.95,option=1)

result
# 
# $causal_effect
# [1] -0.005415744
# 
# $causal_pvalue
# [1] 0.2283658
# 
# $correlated_pleiotropy_effect
# [1] 0.0003259172
# 
# $sigmabeta
# [1] 0.2290256
# 
# $sigmaeta
# [1] 0.02211957
# 
# $sigma_error_1
# [1] 0.8513625
# 
# $sigma_error_2
# [1] 0.9999995

#relaxed (pvalue<10^(-5))
snps_apoe<-res_pseudo_micro19[pvalue<10^(-5)&molecular_trait_id=="APOE"]$variant
length(snps_apoe) #109

#gwas

com_snps_apoe<-res_gwas19[variant%in%snps_apoe]$variant 
length(com_snps_apoe) #77/109

#need LD 
#will used LDlink db first

cat(paste(str_extract(com_snps_apoe,'chr[0-9]+:[0-9]+'),collapse = "\n"))
#extract LD matrix from 
#https://ldlink.nci.nih.gov/?tab=ldmatrix

system("wget -O data/ldlink_snp_apoe_relaxed.txt https://ldlink.nci.nih.gov/tmp/r2_84199.txt")

ld_mat<-fread("data/ldlink_snp_apoe_relaxed.txt")

result_relaxed<-MRAID(res_pseudo_micro19[molecular_trait_id=="APOE"][com_snps_apoe,on="variant"]$beta_z,
              res_gwas19[com_snps_apoe,on="variant"]$beta_z,
              Sigma1sin=as.matrix(data.frame(ld_mat,row.names = "RS_number")),
              Sigma2sin=as.matrix(data.frame(ld_mat,row.names = "RS_number")),
              samplen1=n_eqtl, #411
              samplen2=n_gwas, #486115
              Gibbsnumber=1000,burninproportion=0.2,
              pi_beta_shape=0.5,
              pi_beta_scale=4.5,
              pi_c_shape=0.5,pi_c_scale=9.5,pi_1_shape=0.5,pi_1_scale=1.5,pi_0_shape=0.05,pi_0_scale=9.95,option=1)

result_relaxed
# $causal_effect
# [1] -0.0003997229
# 
# $causal_pvalue
# [1] 0.8812703
# 
# $correlated_pleiotropy_effect
# [1] -0.000474371
# 
# $sigmabeta
# [1] 0.1633833
# 
# $sigmaeta
# [1] 0.01219819
# 
# $sigma_error_1
# [1] 0.6142384
# 
# $sigma_error_2
# [1] 1.000046
#worst than before

#BIN1####
gene<-gene
res_pseudo_micro2<-fread("data/pseudobulk_eqtl/nominal_sumstat/Mic.log2cpm.cov_pca.resid.PEER.cov.2.norminal.cis_long_table.txt.gz")

res_pseudo_micro2[pvalue<10^(-8)&molecular_trait_id==gene] #40

res_pseudo_micro2[pvalue<10^(-5)&molecular_trait_id==gene] #109, ok test this one

#stringent (pvalue<10^(-8))
snps_bin1<-res_pseudo_micro2[pvalue<10^(-8)&molecular_trait_id==gene]$variant

#gwas
res_gwas2<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/ADGWAS_Bellenguez_2022/ADGWAS2022.chr2.sumstat.tsv")
res_gwas2[,variant:=str_replace(variant,'_',':')]
res_gwas2[variant%in%snps_bin1] #35/40

com_snps_bin1<-unique(res_gwas2[variant%in%snps_bin1]$variant )#35/40
length(com_snps_bin1)
?MRAID

#need z score
res_gwas2[,beta_mean:=mean(beta)]
res_gwas2[,beta_sd:=sd(beta)]
res_gwas2[,beta_z:=(beta-beta_mean)/beta_sd]


res_pseudo_micro2[,beta_mean:=mean(beta,na.rm = T),by=.(molecular_trait_id)]
res_pseudo_micro2[,beta_sd:=sd(beta,na.rm = T),by=.(molecular_trait_id)]
res_pseudo_micro2[,beta_z:=(beta-beta_mean)/beta_sd,by=.(molecular_trait_id)]
res_pseudo_micro2[pvalue<10^(-8)&molecular_trait_id==gene]

#need LD 
#will used LDlink db first

cat(paste(str_extract(com_snps_bin1,'chr[0-9]+:[0-9]+'),collapse = "\n"))
#extract LD matrix from 
#https://ldlink.nci.nih.gov/?tab=ldmatrix
#The following RS number(s) or coordinate(s) inputs have warnings: chr2:127091966

system("wget -O data/ldlink_snp_bin1.txt https://ldlink.nci.nih.gov/tmp/r2_48184.txt")

ld_mat<-fread("data/ldlink_snp_bin1.txt")
dim(ld_mat) #34 

com_snps_bin1<-com_snps_bin1[!str_detect(com_snps_bin1,"chr2:127091966")]
length(com_snps_bin1) #34
ld_mat[1:10,1:10]
n_eqtl<-unique(res_pseudo_micro2[variant%in%com_snps_bin1&molecular_trait_id==gene]$n)
res_gwas2[,n_tot:=n_cases+n_controls]
n_gwas<-round(mean(res_gwas2[variant%in%com_snps_bin1]$n_tot))

result<-MRAID(res_pseudo_micro2[molecular_trait_id==gene][com_snps_bin1,on="variant"]$beta_z,
              res_gwas2[com_snps_bin1,on="variant"]$beta_z,
              Sigma1sin=as.matrix(data.frame(ld_mat,row.names = "RS_number")),
              Sigma2sin=as.matrix(data.frame(ld_mat,row.names = "RS_number")),
              samplen1=n_eqtl, #411
              samplen2=n_gwas, #474719
              Gibbsnumber=1000,burninproportion=0.2,
              pi_beta_shape=0.5,
              pi_beta_scale=4.5,
              pi_c_shape=0.5,pi_c_scale=9.5,pi_1_shape=0.5,pi_1_scale=1.5,pi_0_shape=0.05,pi_0_scale=9.95,option=1)

result
# $causal_effect
# [1] 0.000753068
# 
# $causal_pvalue
# [1] 0.8932303
# 
# $correlated_pleiotropy_effect
# [1] 0.0001090178
# 
# $sigmabeta
# [1] 0.06536586
# 
# $sigmaeta
# [1] 0.02862642
# 
# $sigma_error_1
# [1] 1.003891
# 
# $sigma_error_2
# [1] 0.999924




#others eQTL results ?####
res_chr19<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/rna-seq/MSSM/MiGA/TensorQTL_Summary_Statistics/10-02-2022/MiGA.GFM.chr19.norminal.cis_long_table.txt")
res_chr19#n=66
res_chr19[pvalue<10^(-8)] #69 but no APOE

res_chr19[molecular_trait_id=="ENSG00000130203"][order(pvalue)][1:100]


res_chr19<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/rna-seq/MSSM/ROSMAP/TensorQTL_Summary_Statistics/11-28-2022/ROSMAP.chr19.norminal.cis_long_table.txt")
res_chr19#n=225
res_chr19[pvalue<10^(-8)] #4262

res_chr19[pvalue<10^(-6)&molecular_trait_id=="ENSG00000130203"] #7
res_chr19[pvalue<10^(-5)&molecular_trait_id=="ENSG00000130203"] #15


#BIN1 ? ENSG00000136717

res_chr2<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/rna-seq/MSSM/ROSMAP/TensorQTL_Summary_Statistics/11-28-2022/ROSMAP.chr2.norminal.cis_long_table.txt")
res_chr2[pvalue<10^(-8)] #4262
res_chr2[pvalue<10^(-8)&molecular_trait_id=="ENSG00000136717"] #0
res_chr2[molecular_trait_id=="ENSG00000136717"][order(pvalue)][pvalue<10^(-6)] #39


res_chr2[molecular_trait_id=="ENSG00000136717"][order(pvalue)][pvalue<10^(-5)] #66


res_chr2<-fread("../../xQTL/ftp/ftp_fgc_xqtl/projects/rna-seq/BU/ROSMAP_DLPFC/eQTL/association_scan/norminal_qced_files/dlpfc_batch_all.rnaseqc.low_expression_filtered.outlier_removed.tmm.expression.bed.processed_phenotype.per_chrom_dlpfc_batch_all.rnaseqc.ROSMAP_covariates.ROSMAP_NIA_WGS.pca.PEER.txt.2.norminal.cis_long_table.txt")
res_chr2[pvalue<10^(-8)] #120k
res_chr2[pvalue<10^(-8)&molecular_trait_id=="ENSG00000136717"] #0
res_chr2[molecular_trait_id=="ENSG00000136717"][order(pvalue)][pvalue<10^(-6)] #12


res_chr2[molecular_trait_id=="ENSG00000136717"][order(pvalue)][pvalue<10^(-5)] #26


res_chr2[,n.asso:=sum(pvalue<10^(-8)),by="molecular_trait_id"]


trans<-TransEnsembltoSymbol(unique(res_chr2$molecular_trait_id))
res_chr2<-merge(res_chr2[,ensembl_gene_id:=molecular_trait_id],trans)

unique(res_chr2[order(-n.asso)],by="molecular_trait_id")[1:100]

table(res_chr2[order(nmolecular_trait_id)]pvalue<10^(-8)]$molecular_trait_id)

res_chr2[pvalue<10^(-6)&hgnc_symbol=="NCK2"] 



#test causality on list of candidates genes on Microglia pseudobulk####
#genes from ADSP: https://adsp.niagads.org/gvc-top-hits-list/ 
cand_genes_adsp<-fread("xqtl-protocol/data/candidates_causal_genes_adsp.csv")
  
#genes from "Integration of Alzheimer’s disease genetics and myeloid genomics identifies disease risk regulatory elements and genes"
#Novikova et al, Nat comm., 2021 : https://www.nature.com/articles/s41467-021-21823-y/tables/1 
cand_genes_myelo<-fread("xqtl-protocol/data/candidates_causal_genes_myeloid.csv")

#get LD matrix for every SNPs associated to it
cand_genes<-union(cand_genes_adsp$Gene,cand_genes_myelo$gene)

#add chr information
cand_genes_anno<-GetChrInfos(cand_genes)
cand_genes_anno<-cand_genes_anno[chromosome_name%in%c(1:22,"X","Y")]
fwrite(cand_genes_anno,file.path(out,"cand_genes_anno.csv"))
#test pipeline for APOE
g<-"APOE"
message("testing MRAID for")
chr<-cand_genes_anno[hgnc_symbol==g]$chromosome_name
message(g," on chr",chr)

#get eQTL Data
res_eqtl<-fread(paste0("data/pseudobulk_eqtl/nominal_sumstat/Mic.log2cpm.cov_pca.resid.PEER.cov.",chr,".norminal.cis_long_table.txt.gz"))[molecular_trait_id==g]

snps_asso_gene<-res_eqtl[pvalue<10^(-8)]$variant
n_snps_gene<-length(snps_asso_gene)
message(n_snps_gene," SNPs asso to ",g)
if(n_snps_gene>20){
  #get GWAS data
  res_gwas<-fread(paste0("../../xQTL/ftp/ftp_fgc_xqtl/projects/ADGWAS_Bellenguez_2022/ADGWAS2022.chr",chr,".sumstat.tsv"))
  
  res_gwas[,variant:=str_replace(variant,'_',':')]
  comm_snps<-res_gwas[variant%in%snps_asso_gene]$variant
  n_comm_snps<-length(comm_snps) #43/57
  
  message("of which ",n_comm_snps," found in GWAS (",round(n_comm_snps/n_snps_gene*100),"%)")
  
  #calculate z score of beta effects
  res_gwas[,beta_mean:=mean(beta)]
  res_gwas[,beta_sd:=sd(beta)]
  res_gwas[,beta_z:=(beta-beta_mean)/beta_sd]
  
  
  res_eqtl[,beta_mean:=mean(beta,na.rm = T)]
  res_eqtl[,beta_sd:=sd(beta,na.rm = T)]
  res_eqtl[,beta_z:=(beta-beta_mean)/beta_sd]
  
  #get LD  matrix
  #will used LDlink db first
  #access to API thanks to a token saved in .Renviron
  #usethis::edit_r_environ()

  ld_mat<-LDmatrix(str_extract(comm_snps,'chr[0-9]+:[0-9]+'), 
                   pop = "ALL", 
                   r2d = "r2", 
                   token =Sys.getenv("LDLINK_TOKEN"), 
                   file = FALSE,
                   genome_build = "grch38_high_coverage")
  print(head(ld_mat))
  
  #get the rsid - chr coords link
  coords<-str_extract(comm_snps,'[0-9]+:[0-9]+')
  coords<-paste(coords,str_remove(coords,"[0-9]+:"),sep = ":")
  snps_rsid<-GetRSID(coords)
  snps_rsid<-data.table(snps_rsid)
  
  coords_2<-str_remove(str_extract(comm_snps,':[0-9]+'),":")
  snps_rsid[,coord:=comm_snps[chrom_start<=coords_2&chrom_end>=coords_2],by="refsnp_id"]
  snps_rsid[,RS_number:=refsnp_id]
  ld_mat_anno<-merge(data.table(ld_mat),snps_rsid,by="RS_number")
  
  comm_snps2<-intersect(comm_snps,ld_mat_anno$coord)
  length(comm_snps2)
  
  rsids<-ld_mat_anno[comm_snps2,on="coord"]$refsnp_id
  
  #get n sample by test
  n_sample_eqtl<-unique(res_eqtl[variant%in%comm_snps2]$n)
  res_gwas[,n_tot:=n_cases+n_controls]
  n_sample_gwas<-round(mean(res_gwas[variant%in%comm_snps2]$n_tot))

  #run MRAID with option 1 and 2 because outcome GWAS have large sample size (see ?MRAID)
  
  res<-Reduce(rbind,lapply(c(1,2),function(x){
    result<-MRAID(res_eqtl[comm_snps2,on="variant"]$beta_z,
                  res_gwas[comm_snps2,on="variant"]$beta_z,
                  Sigma1sin=as.matrix(data.frame(ld_mat_anno,row.names = "RS_number")[rsids,rsids]),
                  Sigma2sin=as.matrix(data.frame(ld_mat_anno,row.names = "RS_number")[rsids,rsids]),
                  samplen1=n_sample_eqtl, #411
                  samplen2=n_sample_gwas, #486115
                  Gibbsnumber=1000,burninproportion=0.2,
                  pi_beta_shape=0.5,
                  pi_beta_scale=4.5,
                  pi_c_shape=0.5,pi_c_scale=9.5,pi_1_shape=0.5,pi_1_scale=1.5,pi_0_shape=0.05,pi_0_scale=9.95,
                  option=x)
    return( as.data.table(result)[,method:=x])
    
  }))
  
  
  
 
  
}

#on all genes cand
#run MRAID_on_candidates_genes

#on all pseudobulk####

#on bulkRNA-seq ####