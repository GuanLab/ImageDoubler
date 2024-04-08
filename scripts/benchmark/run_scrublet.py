
import scanpy as sc
import scrublet as scr
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


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
    
def detect_and_evaluate(adata):
    mapping = {0: "singlet", 1: "doublet"}
    
    for threshold in np.arange(0.06, 0.22, 0.02):
        identifier = f"doublets_thresh_{threshold:.2f}"
        scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06)
        _, _ = scrub.scrub_doublets()
        predicted_doublets = scrub.call_doublets(threshold=threshold).astype(int)
        adata.obs[f"preds_{identifier}"] = [mapping[x] for x in predicted_doublets]
        
        df_eva = adata.obs[adata.obs["class"].notnull() & (adata.obs["class"] != "missing")]
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

adata_img5 = sc.read_h5ad("scrublet_img5.h5ad")
adata_img11 = sc.read_h5ad("scrublet_img11.h5ad")
adata_img5_noMissing = sc.read_h5ad("scrublet_img5_noMissing.h5ad")
adata_img11_noMissing = sc.read_h5ad("scrublet_img11_noMissing.h5ad")

# detect_and_evaluate(adata_img5)
# detect_and_evaluate(adata_img11)
# detect_and_evaluate(adata_img5_noMissing)
# detect_and_evaluate(adata_img11_noMissing)

re_evaluate(adata_img5)
re_evaluate(adata_img11)
re_evaluate(adata_img5_noMissing)
re_evaluate(adata_img11_noMissing)

print(adata_img5.uns)
adata_img5.write_h5ad("scrublet_img5.h5ad")
print(adata_img11.uns)
adata_img11.write_h5ad("scrublet_img11.h5ad")
print(adata_img5_noMissing.uns)
adata_img5_noMissing.write_h5ad("scrublet_img5_noMissing.h5ad")
print(adata_img11_noMissing.uns)
adata_img11_noMissing.write_h5ad("scrublet_img11_noMissing.h5ad")

