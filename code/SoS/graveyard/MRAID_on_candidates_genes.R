out<-"outputs/01-MRAID_test"
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

#ANALYSIS####
#test causality on list of candidates genes on Microglia pseudobulk####

cand_genes_anno<-fread(file.path(out,"cand_genes_anno.csv"))

cand_genes<-cand_genes_anno$hgnc_symbol

res_mraid_all<-Reduce(function(...)rbind(...,fill=TRUE),lapply(cand_genes, function(g){
  message("testing MRAID for")
  chr<-cand_genes_anno[hgnc_symbol==g]$chromosome_name
  message(g," on chr",chr)
  
  #get eQTL Data
  res_eqtl<-fread(paste0("data/pseudobulk_eqtl/nominal_sumstat/Mic.log2cpm.cov_pca.resid.PEER.cov.",chr,".norminal.cis_long_table.txt.gz"))[molecular_trait_id==g]
  
  snps_asso_gene<-res_eqtl[pvalue<10^(-6)]$variant
  n_snps_gene<-length(snps_asso_gene)
  message(n_snps_gene," SNPs asso to ",g)
  res<-data.table(gene=g,n_snps_asso=n_snps_gene)
  
  if(n_snps_gene>20){
    #get GWAS data
    res_gwas<-fread(paste0("../../xQTL/ftp/ftp_fgc_xqtl/projects/ADGWAS_Bellenguez_2022/ADGWAS2022.chr",chr,".sumstat.tsv"))
    
    res_gwas[,variant:=str_replace(variant,'_',':')]
    comm_snps<-res_gwas[variant%in%snps_asso_gene]$variant
    n_comm_snps<-length(comm_snps) #43/57
    
    res[,n_inter_gwas:=n_comm_snps]
    
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
    message("getting the LD matrix")
    ld_mat<-LDmatrix(str_extract(comm_snps,'chr[0-9]+:[0-9]+'), 
                     pop = "CEU", 
                     r2d = "r2", 
                     token =Sys.getenv("LDLINK_TOKEN"), 
                     file = FALSE,
                     genome_build = "grch38_high_coverage")
    print(head(ld_mat))
    fwrite(ld_mat,file.path(out,paste0("LD_matrix_CEU_snps_asso",g,".csv.gz")))
    
    #get the rsid - chr coords link
    # message("linking snps coords to rsid")
    # 
    # coords<-str_extract(comm_snps,'[0-9]+:[0-9]+')
    # coords<-paste(coords,str_remove(coords,"[0-9]+:"),sep = ":")
    #snps_rsid<-data.table(GetRSID(coords))

    coords_2<-str_remove(str_extract(comm_snps,':[0-9]+'),":")
    snps_rsid[,coord:=comm_snps[chrom_start<=coords_2&chrom_end>=coords_2],by="refsnp_id"]
    snps_rsid[,RS_number:=refsnp_id]
    ld_mat_anno<-merge(data.table(ld_mat),snps_rsid,by="RS_number")
    fwrite(ld_mat_anno,file.path(out,paste0("LD_matrix_anno_snps_asso",g,".csv.gz")))
    
    comm_snps2<-intersect(comm_snps,ld_mat_anno$coord)
    length(comm_snps2)
    res[,n_with_ld:=length(comm_snps2)]
    
    rsids<-ld_mat_anno[comm_snps2,on="coord"]$refsnp_id
    res[,cand_snps:=paste(comm_snps2,collapse = "|")][,rsids:=paste(rsids,collapse = "|")]
    
    #get n sample by test
    n_sample_eqtl<-unique(res_eqtl[variant%in%comm_snps2]$n)
    res_gwas[,n_tot:=n_cases+n_controls]
    n_sample_gwas<-round(mean(res_gwas[variant%in%comm_snps2]$n_tot))
    
    res[,n_sample_eqtl:=n_sample_eqtl]
    res[,n_sample_gwas:=n_sample_gwas]
    
    #run MRAID with option 1 and 2 because outcome GWAS have large sample size (see ?MRAID)
    message("running MRAID")
    
    res_mraid<-Reduce(rbind,lapply(c(1,2),function(x){
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
      print(result)
      return( as.data.table(result)[,method:=x][,gene:=g])
      
    }))
    res<-merge(res,res_mraid,by="gene")
  }
  
  return(res)
  
}))

fwrite(res_mraid_all,file.path(out,"res_mraid_cand_genes_eqtl_pseudobulk_microglia.csv.gz"))
message("Success !")