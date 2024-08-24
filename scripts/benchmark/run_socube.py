
import os
import scanpy as sc
import scrublet as scr
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from glob import glob

import warnings
warnings.filterwarnings('ignore')


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
        adata = adata[(adata.obs["class"] != "missing")]
        # adata = adata[adata.obs["class"].notnull() & (adata.obs["class"] != "missing")]
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
    
def detect_and_evaluate(adata, predicts):
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        predicts["predict_type"] = predicts["predict_score"].apply(lambda x: "doublet" if x > thresh else "singlet")
        adata.obs[f"preds_{str(thresh)}"] = predicts["predict_type"]
        
        df_eva = adata.obs[adata.obs["class"].notnull() & 
                           (adata.obs["class"] != "missing") &
                           (adata.obs["difficult"] == "False")]
        adata.uns[f"eva_{str(thresh)}"] = evaluate(df_eva, "class", f"preds_{str(thresh)}")

adata_img5 = read_counts("../counts/raw_counts_img5_glevel.txt", "Image5", exclude_missing=False)
adata_img5_noMissing = read_counts("../counts/raw_counts_img5_glevel.txt", "Image5", exclude_missing=True)
adata_img11 = read_counts("../counts/raw_counts_img11_glevel.txt", "Image11", exclude_missing=False)
adata_img11_noMissing = read_counts("../counts/raw_counts_img11_glevel.txt", "Image11", exclude_missing=True)

adata_img5.write_h5ad("socube_img5.h5ad")
adata_img5_noMissing.write_h5ad("socube_img5_noMissing.h5ad")
adata_img11.write_h5ad("socube_img11.h5ad")
adata_img11_noMissing.write_h5ad("socube_img11_noMissing.h5ad")

os.system("socube -i ./socube_img5.h5ad -o socube_img5 --gpu-ids 0")
os.system("socube -i ./socube_img5_noMissing.h5ad -o socube_img5_noMissing --gpu-ids 0")
os.system("socube -i ./socube_img11.h5ad -o socube_img11 --gpu-ids 0")
os.system("socube -i ./socube_img11_noMissing.h5ad -o socube_img11_noMissing --gpu-ids 0")


preds = pd.read_csv(glob("socube_img5/outputs/*/final_result_0.5.csv")[0], index_col=0)
detect_and_evaluate(adata_img5, preds)

preds = pd.read_csv(glob("socube_img5_noMissing/outputs/*/final_result_0.5.csv")[0], index_col=0)
detect_and_evaluate(adata_img5_noMissing, preds)

preds = pd.read_csv(glob("socube_img11/outputs/2*/final_result_0.5.csv")[0], index_col=0)
detect_and_evaluate(adata_img11, preds)

preds = pd.read_csv(glob("socube_img11_noMissing/outputs/*/final_result_0.5.csv")[0], index_col=0)
detect_and_evaluate(adata_img11_noMissing, preds)

adata_img5.write_h5ad("socube_img5.h5ad")
adata_img5_noMissing.write_h5ad("socube_img5_noMissing.h5ad")
adata_img11.write_h5ad("socube_img11.h5ad")
adata_img11_noMissing.write_h5ad("socube_img11_noMissing.h5ad")