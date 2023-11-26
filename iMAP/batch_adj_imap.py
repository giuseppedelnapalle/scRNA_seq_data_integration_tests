#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correct for batch effects using iMAP
datasets pbmc_10k_3p and pbmc_10k_5p
"""

# =============================================================================

# 1 set up Python session

# set working directory
import os
wd = '/home/nikola/Project_Data/Python_data/Spyder/batch_adj/imap'
os.chdir(wd)

# import packages
import scanpy as sc
import imap
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
# dir_p = os.path.join(dir_o, 'plots')
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

adata = ad.AnnData(mat_vs.toarray())

# add obs_names
adata.obs_names = anndt.obs_names.append(anndt_2.obs_names)

# add var_names
adata.var_names = c_genes

# add meta data
adata.obs['batch'] = ([0] * anndt.shape[0]) + ([1] * anndt_2.shape[0])
adata.obs['cell_type'] = [1] * adata.shape[0]
celltype = adata.obs['cell_type']

# Preprocessing
# adata = imap.stage1.data_preprocess(adata, 'batch')  # Preprocess the data.(count data -> log-format data, high-var genes selected)
adata = imap.stage1.data_preprocess(adata, 'batch', n_top_genes=adata.shape[1])
adata  # Output the basic information of the preprocessed data.

# Batch effect removal by iMAP
### Stage I
"""
n_epochs: Number of epochs (in Stage I) to train the model. 
It has a great effect on the results of iMAP. 
The number of epochs should be set according to the number of cells in your dataset. 
For example, 150 epochs is generally fine for around or greater than 10,000 cells. 
100 epochs or fewer for fewer than 5,000 cells
"""
begin = time.time()
EC, ec_data = imap.stage1.iMAP_fast(adata, key="batch", n_epochs=100) 
end = time.time()
print(f"Total runtime of the program is {end - begin}.")
# 5567.751549959183

### Stage II
begin = time.time()
output_results = imap.stage2.integrate_data(adata, ec_data, inc = False, n_epochs=80)
# TypeError: sequence item 0: expected str instance, int found (fixed)
end = time.time()
print(f"Total runtime of the program is {end - begin}.")
# 1224.9830780029297

type(output_results)
output_results.shape
output_results.mean(axis=1).mean()
output_results[:10,:10]

# write csv files
n_c_ds = (adata.obs.batch==0).sum()
output_results[:n_c_ds,:].mean(axis=1).mean()
# 0.2506803938857236
mat_c = pd.DataFrame(output_results[:n_c_ds,:], columns=adata.var_names, 
                      index=adata.obs_names[:n_c_ds])
# mat_c.head()
# mat_c.shape
fn_csv = os.path.join(dir_o, 'mat_3p_cd14_mono_imap_corrected.csv')
mat_c.to_csv(fn_csv)

output_results[n_c_ds:adata.shape[0],:].mean(axis=1).mean()
# 0.18270284365060013
mat_c_2 = pd.DataFrame(output_results[n_c_ds:adata.shape[0],:], columns=adata.var_names, 
                      index=adata.obs_names[n_c_ds:adata.shape[0]])
# mat_c_2.shape
fn_csv_2 = os.path.join(dir_o, 'mat_5p_cd14_mono_imap_corrected.csv')
mat_c_2.to_csv(fn_csv_2)

# Visualizations
import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

#### UMAP ####
def data2umap(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    embedding_ = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        n_components = 2,
        learning_rate = 1.0,
        spread = 1.0,
        set_op_mix_ratio = 1.0,
        local_connectivity = 1,
        repulsion_strength = 1,
        negative_sample_rate = 5,
        angular_rp_forest = False,
        verbose = False
    ).fit_transform(embedding)
    return embedding_

def umap_plot(data, hue, title, save_path):
    # import seaborn as sns
    fig = sns.lmplot(
        x = 'UMAP_1',
        y = 'UMAP_2',
        data = data,
        fit_reg = False,
        legend = True,
        # size = 9, # TypeError: lmplot() got an unexpected keyword argument 'size'
        hue = hue,
        scatter_kws = {'s':4, "alpha":0.6}
    )
    plt.title(title, weight='bold').set_fontsize('20')
    fig.savefig(save_path)
    plt.close()

def gplot(embedding_, batch_info, celltype_info, filename):
    test = pd.DataFrame(embedding_, columns=['UMAP_1', 'UMAP_2'])
    test['Label1'] = batch_info
    test['Label2'] = celltype_info
    title = f' '
    for i in range(1,3):
        hue = f'Label{i}'
        save_path = './pic/'+filename + f'{i}.png'
        umap_plot(test, hue, title, save_path)

# Visualizations for the representations from stage I
embedding_ = data2umap(np.array(ec_data.X), n_pca=30)
gplot(embedding_, np.array(ec_data.obs['batch']), np.array([celltype[item] for item in ec_data.obs_names]), 'cellline_ec_')

# Visualizations for the final output results
embedding_ = data2umap(output_results, n_pca=30)
gplot(embedding_, np.array(adata.obs['batch']), np.array([celltype[item] for item in adata.obs_names]), 'cellline_G_')

# =============================================================================
