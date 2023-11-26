#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correct for batch effects using Scanorama
datasets pbmc_10k_3p and pbmc_10k_5p
"""


# =============================================================================

# 1 set up Python session

# set working directory
import os
wd = '/home/nikola/Project_Data/Python_data/Spyder/batch_adj/scanorama'
os.chdir(wd)

# import packages
import scanpy as sc
import scanorama
import time
# import anndata
# import numpy as np
import pandas as pd
# from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')

# input directories
dir_i = '/home/nikola/Project_Data/R_data/tests/ds_integration/input'
dir_i_2 = '/home/nikola/Project_Data/R_data/tests/ds_integration/output/cell_annotations/pbmc_10k_3p'
dir_i_3 = '/home/nikola/Project_Data/R_data/tests/ds_integration/output/cell_annotations/pbmc_10k_5p'

# input files
fn_dt = '/'.join((dir_i, '3p_pbmc10k_filt.h5'))
fn_mt = '/'.join((dir_i_2, 'c_ann_pbmc10k_3p.csv'))
fn_dt_2 = '/'.join((dir_i, '5p_pbmc10k_filt.h5'))
fn_mt_2 = '/'.join((dir_i_3, 'c_ann_pbmc10k_5p.csv'))

# output directory
dir_o = os.path.join(wd, 'output')
# os.mkdir(dir_o)

# plot directory
dir_p = os.path.join(dir_o, 'plots')
# os.mkdir(dir_p)

# cell identity
c_ident = 'cd14_mono'

# =============================================================================


# =============================================================================

# 2 preprocess data

# 2.1 dataset 1
# create AnnData object
anndt_o = sc.read_10x_h5(fn_dt)

# load meta
c_ann = pd.read_csv(fn_mt, header=None)
# c_ann.rename(columns = {0:'barcode', 1:'c_ident'}, inplace = True)

# select cells with specified cell annotation
c_ann_f = c_ann.loc[c_ann.iloc[:,1] == c_ident]
keep = anndt_o.obs.index.isin(c_ann_f.iloc[:,0])
anndt = anndt_o[keep,:]
anndt.shape

# make var names unique
anndt.var_names_make_unique()
anndt

# Show those genes that yield the highest fraction of counts in each single cell, across all cells
sc.pl.highest_expr_genes(anndt, n_top=20, show=False)
plt.savefig(os.path.join(dir_p, 'highest_expr_genes_raw_counts.pdf'), bbox_inches='tight')
plt.close()

# basic filtering
sc.pp.filter_cells(anndt, min_genes=200)
sc.pp.filter_genes(anndt, min_cells=3)

# percentage of mitochondrial genes
anndt.var['mt'] = anndt.var_names.str.startswith('MT-')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt, qc_vars=['mt'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt.obs.pct_counts_mt.median()
# histogram
plt.hist(anndt.obs.pct_counts_mt, bins=50, density=False, alpha=.5, histtype='stepfilled', 
          color='steelblue', edgecolor='none')
plt.savefig(os.path.join(dir_p, 'histogram_pct_counts_mt_raw_counts.pdf'))
plt.close()
# density plot
sns.kdeplot(anndt.obs.pct_counts_mt, fill=True, alpha=.5, linewidth=0)
plt.savefig(os.path.join(dir_p, 'density_plot_pct_counts_mt_raw_counts.pdf'))
plt.close()

# percentage of ribosomal genes
anndt.var['rb'] = anndt.var_names.str.match('RP[SL]')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt, qc_vars=['rb'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt.obs.pct_counts_rb.median()
# histogram
plt.hist(anndt.obs.pct_counts_rb, bins=50, density=False, alpha=.5, histtype='stepfilled', 
          color='steelblue', edgecolor='none')
plt.savefig(os.path.join(dir_p, 'histogram_pct_counts_rb_raw_counts.pdf'))
plt.close()
# density plot
sns.kdeplot(anndt.obs.pct_counts_rb, fill=True, alpha=.5, linewidth=0)
plt.savefig(os.path.join(dir_p, 'density_plot_pct_counts_rb_raw_counts.pdf'))
plt.close()

# violin plot
sc.pl.violin(anndt, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_rb'],
             jitter=0.4, multi_panel=True, show=False)
plt.savefig(os.path.join(dir_p, 'violin_plot_qc_metrics_raw_counts.pdf'))
plt.close()

# scatter plots
# pct_counts_mt vs total_counts
sc.pl.scatter(anndt, x='total_counts', y='pct_counts_mt', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_mt_n_counts_raw_counts.pdf'))
plt.close()
# n_genes_by_counts vs total_counts
sc.pl.scatter(anndt, x='total_counts', y='n_genes_by_counts', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_n_genes_n_counts_raw_counts.pdf'))
plt.close()
# pct_counts_rb vs total_counts
sc.pl.scatter(anndt, x='total_counts', y='pct_counts_rb', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_rb_n_counts_raw_counts.pdf'))
plt.close()
# pct_counts_rb vs pct_counts_mt
sc.pl.scatter(anndt, x='pct_counts_mt', y='pct_counts_rb', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_rb_pct_mt_raw_counts.pdf'))
plt.close()
# log1p
# pct_counts_mt vs log1p_total_counts
sc.pl.scatter(anndt, x='log1p_total_counts', y='pct_counts_mt', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_mt_log1p_n_counts_raw_counts.pdf'))
plt.close()
# log1p_n_genes_by_counts vs log1p_total_counts
sc.pl.scatter(anndt, x='log1p_total_counts', y='log1p_n_genes_by_counts', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_log1p_n_genes_log1p_n_counts_raw_counts.pdf'))
plt.close()
# pct_counts_rb vs total_counts
sc.pl.scatter(anndt, x='log1p_total_counts', y='pct_counts_rb', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_rb_log1p_n_counts_raw_counts.pdf'))
plt.close()

# filtering
# n_genes_by_counts, aka the number of genes with at least 1 count in a cell
anndt.obs.n_genes_by_counts
flt = (anndt.obs.n_genes_by_counts > 500) & (anndt.obs.n_genes_by_counts < 5000) & (anndt.obs.pct_counts_mt < 12)
flt.sum()
anndt = anndt[flt, :]

# Total-count normalize (library-size correct) the data matrix X
sc.pp.normalize_total(anndt, target_sum=1e4)
# anndt.layers['lib_size_norm'] = anndt.X

# logarithmize the data
sc.pp.log1p(anndt)
# anndt.layers['log1p_norm'] = anndt.X

# Identify highly-variable genes
sc.pp.highly_variable_genes(anndt, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(anndt, show=False)
plt.savefig(os.path.join(dir_p, 'highly_variable_genes_log1p_norm.pdf'))
plt.close()

# 2.2 dataset 2

# create AnnData object
anndt_2_o_2 = sc.read_10x_h5(fn_dt_2)

# load meta
c_ann_2 = pd.read_csv(fn_mt_2, header=None)

# select cells with specified cell annotation
c_ann_2_f = c_ann_2.loc[c_ann_2.iloc[:,1] == c_ident]
keep_2 = anndt_2_o_2.obs.index.isin(c_ann_2_f.iloc[:,0])
anndt_2 = anndt_2_o_2[keep_2,:]
anndt_2.shape

# make var names unique
anndt_2.var_names_make_unique()
anndt_2

# Show those genes that yield the highest fraction of counts in each single cell, across all cells
sc.pl.highest_expr_genes(anndt_2, n_top=20, show=False)
plt.savefig(os.path.join(dir_p, 'highest_expr_genes_raw_counts.pdf'), bbox_inches='tight')
plt.close()

# basic filtering
sc.pp.filter_cells(anndt_2, min_genes=200)
sc.pp.filter_genes(anndt_2, min_cells=3)

# percentage of mitochondrial genes
anndt_2.var['mt'] = anndt_2.var_names.str.startswith('MT-')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt_2, qc_vars=['mt'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt_2.obs.pct_counts_mt.median()
# histogram
plt.hist(anndt_2.obs.pct_counts_mt, bins=50, density=False, alpha=.5, histtype='stepfilled', 
          color='steelblue', edgecolor='none')
plt.savefig(os.path.join(dir_p, 'histogram_pct_counts_mt_raw_counts.pdf'))
plt.close()
# density plot
sns.kdeplot(anndt_2.obs.pct_counts_mt, fill=True, alpha=.5, linewidth=0)
plt.savefig(os.path.join(dir_p, 'density_plot_pct_counts_mt_raw_counts.pdf'))
plt.close()

# percentage of ribosomal genes
anndt_2.var['rb'] = anndt_2.var_names.str.match('RP[SL]')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt_2, qc_vars=['rb'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt_2.obs.pct_counts_rb.median()
# histogram
plt.hist(anndt_2.obs.pct_counts_rb, bins=50, density=False, alpha=.5, histtype='stepfilled', 
          color='steelblue', edgecolor='none')
plt.savefig(os.path.join(dir_p, 'histogram_pct_counts_rb_raw_counts.pdf'))
plt.close()
# density plot
sns.kdeplot(anndt_2.obs.pct_counts_rb, fill=True, alpha=.5, linewidth=0)
plt.savefig(os.path.join(dir_p, 'density_plot_pct_counts_rb_raw_counts.pdf'))
plt.close()

# violin plot
sc.pl.violin(anndt_2, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_rb'],
             jitter=0.4, multi_panel=True, show=False)
plt.savefig(os.path.join(dir_p, 'violin_plot_qc_metrics_raw_counts.pdf'))
plt.close()

# scatter plots
# pct_counts_mt vs total_counts
sc.pl.scatter(anndt_2, x='total_counts', y='pct_counts_mt', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_mt_n_counts_raw_counts.pdf'))
plt.close()
# n_genes_by_counts vs total_counts
sc.pl.scatter(anndt_2, x='total_counts', y='n_genes_by_counts', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_n_genes_n_counts_raw_counts.pdf'))
plt.close()
# pct_counts_rb vs total_counts
sc.pl.scatter(anndt_2, x='total_counts', y='pct_counts_rb', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_rb_n_counts_raw_counts.pdf'))
plt.close()
# pct_counts_rb vs pct_counts_mt
sc.pl.scatter(anndt_2, x='pct_counts_mt', y='pct_counts_rb', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_rb_pct_mt_raw_counts.pdf'))
plt.close()
# log1p
# pct_counts_mt vs log1p_total_counts
sc.pl.scatter(anndt_2, x='log1p_total_counts', y='pct_counts_mt', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_mt_log1p_n_counts_raw_counts.pdf'))
plt.close()
# log1p_n_genes_by_counts vs log1p_total_counts
sc.pl.scatter(anndt_2, x='log1p_total_counts', y='log1p_n_genes_by_counts', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_log1p_n_genes_log1p_n_counts_raw_counts.pdf'))
plt.close()
# pct_counts_rb vs total_counts
sc.pl.scatter(anndt_2, x='log1p_total_counts', y='pct_counts_rb', show=False)
plt.savefig(os.path.join(dir_p, 'scatter_plot_pct_rb_log1p_n_counts_raw_counts.pdf'))
plt.close()

# filtering
# n_genes_by_counts, aka the number of genes with at least 1 count in a cell
anndt_2.obs.n_genes_by_counts
flt = (anndt_2.obs.n_genes_by_counts > 500) & (anndt_2.obs.n_genes_by_counts < 5000) & (anndt_2.obs.pct_counts_mt < 8)
flt.sum()
anndt_2 = anndt_2[flt, :]

# Total-count normalize (library-size correct) the data matrix X
sc.pp.normalize_total(anndt_2, target_sum=1e4)
# anndt_2.layers['lib_size_norm'] = anndt_2.X

# logarithmize the data
sc.pp.log1p(anndt_2)
# anndt_2.layers['log1p_norm'] = anndt_2.X

# Identify highly-variable genes
sc.pp.highly_variable_genes(anndt_2, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(anndt_2, show=False)
plt.savefig(os.path.join(dir_p, 'highly_variable_genes_log1p_norm.pdf'))
plt.close()

# =============================================================================


# =============================================================================

# 3 correct for batch effect using Scanorama

anndt_lst = [anndt, anndt_2]

# Integration and batch correction
begin = time.time()
corrected = scanorama.correct_scanpy(anndt_lst, return_dimred=True)
end = time.time()
print(f"Total runtime of the program is {end - begin}.")
# 13.844042778015137

# export batch corrected data
fn_o = os.path.join(dir_o, 'pbmc_10k_3p_cd14_mono_corrected.h5ad')
corrected[0].write(fn_o, compression='gzip')

fn_o_2 = os.path.join(dir_o, 'pbmc_10k_5p_cd14_mono_corrected.h5ad')
corrected[1].write(fn_o_2, compression='gzip')

print(anndt.X)
print(corrected[0].X)
print(anndt_2.X)
print(corrected[1].X)
# negative values found

# export as csv files
corrected_dt = sc.read_h5ad(fn_o)
corrected_dt
type(corrected_dt.X)
# scipy.sparse._csr.csr_matrix

mat_c = pd.DataFrame(corrected_dt.X.toarray(), columns=corrected_dt.var_names, 
                     index=corrected_dt.obs_names)
mat_c.head()
mat_c.shape
fn_csv = os.path.join(dir_o, 'mat_3p_cd14_mono_scanorama_corrected.csv')
mat_c.to_csv(fn_csv)

idx = 500
mat_c.iloc[idx,:][mat_c.iloc[idx,:] == 0].size
# 2

corrected_dt_2 = sc.read_h5ad(fn_o_2)
mat_c_2 = pd.DataFrame(corrected_dt_2.X.toarray(), columns=corrected_dt_2.var_names, 
                     index=corrected_dt_2.obs_names)
mat_c_2.shape
fn_csv_2 = os.path.join(dir_o, 'mat_5p_cd14_mono_scanorama_corrected.csv')
mat_c_2.to_csv(fn_csv_2)

mat_c_2.iloc[idx,:][mat_c_2.iloc[idx,:] == 0].size
# 13627

# =============================================================================
