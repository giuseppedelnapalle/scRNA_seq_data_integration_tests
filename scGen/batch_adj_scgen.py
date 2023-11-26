#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correct for batch effects using scGen
datasets pbmc_10k_3p and pbmc_10k_5p
"""

# =============================================================================

# 1 set up Python session

# set working directory
import os
wd = '/home/nikola/Project_Data/Python_data/Spyder/batch_adj/scgen'
os.chdir(wd)

# import packages
import scanpy as sc
import scgen
import anndata as ad
import time
# import anndata
# import numpy as np
import pandas as pd
import scipy.sparse as sp
# from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns

# matplotlib.use('TkAgg')

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

# basic filtering
sc.pp.filter_cells(anndt, min_genes=200)
sc.pp.filter_genes(anndt, min_cells=3)

# percentage of mitochondrial genes
anndt.var['mt'] = anndt.var_names.str.startswith('MT-')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt, qc_vars=['mt'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt.obs.pct_counts_mt.median()

# percentage of ribosomal genes
anndt.var['rb'] = anndt.var_names.str.match('RP[SL]')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt, qc_vars=['rb'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt.obs.pct_counts_rb.median()

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

# basic filtering
sc.pp.filter_cells(anndt_2, min_genes=200)
sc.pp.filter_genes(anndt_2, min_cells=3)

# percentage of mitochondrial genes
anndt_2.var['mt'] = anndt_2.var_names.str.startswith('MT-')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt_2, qc_vars=['mt'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt_2.obs.pct_counts_mt.median()


# percentage of ribosomal genes
anndt_2.var['rb'] = anndt_2.var_names.str.match('RP[SL]')
# .X used for calculate_qc_metrics
sc.pp.calculate_qc_metrics(anndt_2, qc_vars=['rb'], percent_top=None, use_raw=False, log1p=True, inplace=True)
# anndt_2.obs.pct_counts_rb.median()

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

# =============================================================================


# =============================================================================

# 3 correct for batch effect using scGen

# match common genes
c_genes = [x for x in anndt.var_names if x in anndt_2.var_names]
flt = [(x in c_genes) for x in anndt.var_names]
anndt.var_names[flt]
mat = anndt.X[:, flt]
flt_2 = [(x in c_genes) for x in anndt_2.var_names]
# check if the orders of two gene arrays are the same
(anndt.var_names[flt] == anndt_2.var_names[flt_2]).sum() == len(c_genes)
mat_2 = anndt_2.X[:, flt_2]
mat.shape
mat_2.shape

# create AnnData object for merged data
mat_vs = sp.vstack((mat, mat_2))
mat_vs.shape
train = ad.AnnData(mat_vs)

# add obs_names
train.obs_names = anndt.obs_names.append(anndt_2.obs_names)

# add var_names
train.var_names = c_genes

# add meta data
train.obs['batch'] = ([0] * anndt.shape[0]) + ([1] * anndt_2.shape[0])
train.obs['cell_type'] = [1] * train.shape[0]

# UMAP plot
sc.pp.pca(train)
sc.pp.neighbors(train)
sc.tl.leiden(train)
sc.tl.paga(train)
sc.pl.paga(train, plot=False)
sc.tl.umap(train, init_pos='paga')
sc.pl.umap(train, color=["batch", "cell_type"], wspace=.5, frameon=False, save='_initial.pdf')

# Preprocessing Data
scgen.SCGEN.setup_anndata(train, batch_key="batch", labels_key="cell_type")

# Creating and Saving the model
model = scgen.SCGEN(train)
model.save(os.path.join(dir_o, 'model_batch_removal.pt'), overwrite=True)

# Training the Model
begin = time.time()
model.train(
    max_epochs=100,
    batch_size=32,
    early_stopping=True,
    early_stopping_patience=25,
)
end = time.time()
print(f"Total runtime of the program is {end - begin}.") # est .5 h
# GPU available: False, used: False
# TPU available: False, using: 0 TPU cores
# IPU available: False, using: 0 IPUs
# HPU available: False, using: 0 HPUs
# Epoch 29/100:  29%|██▉       | 29/100 [08:37<21:06, 17.83s/it, loss=789, v_num=1]
# Monitored metric elbo_validation did not improve in the last 25 records. Best score: 2203.296. Signaling Trainer to stop.
# 517.2146480083466

# Batch-Removal
corrected_adata = model.batch_removal()
corrected_adata
# negative values found

corrected_adata.X.mean(axis=0).mean() # mean column sums
# 0.17695135
corrected_adata.X.mean(axis=1).mean() # mean row sums
# 0.17695133

# Visualization of the corrected gene expression data
sc.pp.pca(corrected_adata)
sc.pp.neighbors(corrected_adata)
sc.tl.umap(corrected_adata)
sc.pl.umap(corrected_adata, color=['batch', 'cell_type'], wspace=0.4, frameon=False, save='_corrected.pdf')

# We can also use low-dim corrected gene expression data
sc.pp.neighbors(corrected_adata, use_rep="corrected_latent")
sc.tl.umap(corrected_adata)
sc.pl.umap(corrected_adata, color=['batch', 'cell_type'], wspace=0.4, frameon=False, save='_low_dim_corrected.pdf')

# export batch corrected data
fn_o = os.path.join(dir_o, 'pbmc_10k_3p_5p_cd14_mono_scgen_corrected.h5ad')
corrected_adata.write(fn_o, compression='gzip')

corrected_adata = sc.read_h5ad(fn_o)
# write csv files
n_c_ds = (corrected_adata.obs.batch==0).sum()
mat_c = pd.DataFrame(corrected_adata.X[:n_c_ds,:], columns=corrected_adata.var_names, 
                     index=corrected_adata.obs_names[:n_c_ds])
mat_c.head()
mat_c.shape
fn_csv = os.path.join(dir_o, 'mat_3p_cd14_mono_scgen_corrected.csv')
mat_c.to_csv(fn_csv)

mat_c_2 = pd.DataFrame(corrected_adata.X[n_c_ds:corrected_adata.shape[0],:], columns=corrected_adata.var_names, 
                     index=corrected_adata.obs_names[n_c_ds:corrected_adata.shape[0]])
mat_c_2.shape
fn_csv_2 = os.path.join(dir_o, 'mat_5p_cd14_mono_scgen_corrected.csv')
mat_c_2.to_csv(fn_csv_2)

# =============================================================================
