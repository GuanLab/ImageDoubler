
import scanpy as sc
import scvi
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, balanced_accuracy_score, f1_score)
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore", message=".*within the support of the distribution.*")


targets = pd.read_csv("../../../crop_target/targets.csv")
targets["image_set"] = targets["image_id"].apply(lambda x: x.split("_")[0])
targets["image_pos"] = targets["image_id"].\
    apply(lambda x: f"COL{int(x.split('_')[2]):02d}_ROW{int(x.split('_')[1]):02d}")
targets["class"] = targets["class"].str.lower()
targets["difficult"] = targets["difficult"].fillna(False).astype(str)

def read_counts(the_file, image_set, exclude_missing=False):
    adata = sc.read_csv(the_file, delimiter="\t").T
    adata.obs["image_pos"] = adata.obs.index
    adata.obs = pd.merge(
        adata.obs,
        targets[targets["image_set"] == image_set],
        on='image_pos', 
        how="left").set_index("image_pos")
    adata.obs["class"] = adata.obs["class"].str.lower()
    if exclude_missing:
        adata = adata[(adata.obs["class"] != "missing")].copy()
        # adata = adata[adata.obs["class"].notnull() & (adata.obs["class"] != "missing")].copy()
    return adata

def evaluate(df_eva, gt_col, pred_col):
    try:
        accuracy = accuracy_score(df_eva[gt_col], df_eva[pred_col])
        f1 = f1_score(df_eva[gt_col], df_eva[pred_col], pos_label="doublet")
        confusions = confusion_matrix(df_eva[gt_col], df_eva[pred_col], labels=["singlet", "doublet"])
        tn, fp, fn, tp = confusions.ravel()
        
    except ValueError:
        accuracy, f1, tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "tn": tn, 
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    

def detect_and_evaluate(adata, min_cells=None, n_top_genes=None):
    identifier = f"min_cells_{0 if min_cells is None else min_cells}_n_top_genes_{'all' if n_top_genes is None else n_top_genes}"
    if min_cells is not None:
        sc.pp.filter_cells(adata, min_genes=min_cells)
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True, flavor="seurat_v3")
        
    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train();
    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train();
    
    df = solo.predict()
    df["prediction"] = solo.predict(soft=False)
    adata.obs[f"preds_{identifier}"] = df["prediction"]
    
    df_eva = adata.obs[adata.obs["class"].notnull() & (adata.obs["class"] != "missing")]
    # print(df_eva)
    # print(df_eva[f"preds_{identifier}"])
    # print(df_eva["class"])
    adata.uns[f"eva_{identifier}"] = evaluate(df_eva, "class", f"preds_{identifier}")
    
def re_evaluate(adata):
    # to remove the difficults
    df_eva = adata.obs[adata.obs["class"].notnull() & 
                       (adata.obs["class"] != "missing") &
                       (adata.obs["difficult"] == "False")]
    pred_cols = [col for col in adata.obs.columns if col.startswith("preds_")]
    for col in pred_cols:
        identifier = col.replace("preds_", "")
        adata.uns[f"eva_{identifier}"] = evaluate(df_eva, "class", f"preds_{identifier}")


# adata_img5 = read_counts("../counts/raw_counts_img5_glevel.txt", "Image5", exclude_missing=False)
# adata_img5_noMissing = read_counts("../counts/raw_counts_img5_glevel.txt", "Image5", exclude_missing=True)
# adata_img11 = read_counts("../counts/raw_counts_img11_glevel.txt", "Image11", exclude_missing=False)
# adata_img11_noMissing = read_counts("../counts/raw_counts_img11_glevel.txt", "Image11", exclude_missing=True)

adata_img5 = sc.read_h5ad("solo_img5.h5ad")
adata_img11 = sc.read_h5ad("solo_img11.h5ad")
adata_img5_noMissing = sc.read_h5ad("solo_img5_noMissing.h5ad")
adata_img11_noMissing = sc.read_h5ad("solo_img11_noMissing.h5ad")

# detect_and_evaluate(adata_img5);
# detect_and_evaluate(adata_img5, 1, 2000);
# for min_cells in [None, 1, 10]:
#     for n_top_genes in [None, 2000, 5000]:
#         detect_and_evaluate(adata_img5, min_cells, n_top_genes)
#         detect_and_evaluate(adata_img11, min_cells, n_top_genes)
        
# for min_cells in [None, 1, 10]:
#     for n_top_genes in [None, 2000, 5000]:
#         detect_and_evaluate(adata_img5_noMissing, min_cells, n_top_genes)
#         detect_and_evaluate(adata_img11_noMissing, min_cells, n_top_genes)

re_evaluate(adata_img5)
re_evaluate(adata_img11)
re_evaluate(adata_img5_noMissing)
re_evaluate(adata_img11_noMissing)

print(adata_img5.uns)
adata_img5.write_h5ad("solo_img5.h5ad")
print(adata_img11.uns)
adata_img11.write_h5ad("solo_img11.h5ad")
print(adata_img5_noMissing.uns)
adata_img5_noMissing.write_h5ad("solo_img5_noMissing.h5ad")
print(adata_img11_noMissing.uns)
adata_img11_noMissing.write_h5ad("solo_img11_noMissing.h5ad")
    
    
    
    
