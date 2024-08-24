
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, balanced_accuracy_score, f1_score)
from collections import Counter


def count_cells(detection_result_file, thresh):
    res_file = open(detection_result_file, "r")
    detected_cells = [line.strip().split() for line in res_file]
    n_valid_cell = len([cell for cell in detected_cells if float(cell[1]) > thresh])
    
    if n_valid_cell == 0:
        return (n_valid_cell, "Missing")
    elif n_valid_cell == 1:
        return (n_valid_cell, "Singlet")
    elif n_valid_cell >= 2:
        return (n_valid_cell, "Doublet")
    else:
        raise ValueError(f"Invalid cell number: {n_valid_cell}")
    
def vote(pred_classes: list):
    class_order = {"Doublet": 0, "Singlet": 1, "Missing": 2}
    counter = Counter(pred_classes)
    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], class_order[item[0]]))
    return sorted_items[0][0]

# remove 6 if using labeler 2's targets
image_sets = ["Image" + str(i) for i in [1,2,3,5,6,7,8,9,10,11]]
N = len(image_sets)
cm_all = np.zeros((3, 3))

for image_set in image_sets:
    targets = pd.read_csv("../data/crop_target/labeler1_targets.csv")
    targets["image_set"] = targets["image_id"].apply(lambda x: x.split("_")[0])
    targets = targets[(targets.image_set == image_set) & ~targets.difficult]

    preds_class = []
    for idx, target in targets.iterrows():
        image_id = target["image_id"]
        # the 0.7 is determined by comparing the average of the performances in different threshold
        pred_classes = []
        for model in ["model1", "model2", "model3", "model4", "model5"]:
            pred_n_cells, pred_class = count_cells(f"map_out/loocv/{image_set}/detection-results-{model}/{image_id}.txt", 0.7)
            pred_classes.append(pred_class)
        preds_class.append(vote(pred_classes))
        
    targets["preds_class"] = preds_class

    # preds
    targets.to_csv(f"map_out/loocv/{image_set}/results.csv", index=False)

    # confusion matrix
    cm = confusion_matrix(targets["class"], targets["preds_class"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Doublet", "Missing", "Singlet"])
    disp.plot()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Prediction", fontsize=14)
    plt.ylabel("Ground-truths", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"map_out/loocv/{image_set}/confusion_matrix.png", dpi=150)
    plt.close()

    cm_norm = confusion_matrix(targets["class"], targets["preds_class"], normalize="true")
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=["Doublet", "Missing", "Singlet"])
    disp.plot()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Prediction", fontsize=14)
    plt.ylabel("Ground-truths", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"map_out/loocv/{image_set}/confusion_matrix_norm.png", dpi=150)
    plt.close()
    
    cm_all += cm
    
cm_all_norm = cm_all / cm_all.sum(axis=1, keepdims=True)

disp = ConfusionMatrixDisplay(cm_all, display_labels=["Doublet", "Missing", "Singlet"])
disp.plot()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Prediction", fontsize=14)
plt.ylabel("Ground-truths", fontsize=14)
plt.tight_layout()
plt.savefig(f"map_out/loocv/confusion_matrix_all.png", dpi=150)
plt.close()

disp = ConfusionMatrixDisplay(cm_all_norm, display_labels=["Doublet", "Missing", "Singlet"])
disp.plot()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Prediction", fontsize=14)
plt.ylabel("Ground-truths", fontsize=14)
plt.tight_layout()
plt.savefig(f"map_out/loocv/confusion_matrix_all_norm.png", dpi=150)
plt.close()


# for expression
targets = pd.read_csv("../data/crop_target/labeler1_targets.csv")
targets["image_set"] = targets["image_id"].apply(lambda x: x.split("_")[0])
targets = targets[(targets.image_set.isin(["Image5", "Image11"])) & ~targets.difficult]

preds_class = []
for idx, target in targets.iterrows():
    image_id = target["image_id"]
    # the 0.7 is determined by comparing the average of the performances in different threshold
    pred_classes = []
    for model in ["model1", "model2", "model3", "model4", "model5"]:
        pred_n_cells, pred_class = count_cells(f"map_out/for_expression/detection-results-{model}/{image_id}.txt", 0.7)
        pred_classes.append(pred_class)
    preds_class.append(vote(pred_classes))
    
targets["preds_class"] = preds_class

# preds
targets.to_csv(f"map_out/for_expression/results.csv", index=False)

# confusion matrix
cm = confusion_matrix(targets["class"], targets["preds_class"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Doublet", "Missing", "Singlet"])
disp.plot()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Prediction", fontsize=14)
plt.ylabel("Ground-truths", fontsize=14)
plt.tight_layout()
plt.savefig(f"map_out/for_expression/confusion_matrix.png", dpi=150)
plt.close()

cm_norm = confusion_matrix(targets["class"], targets["preds_class"], normalize="true")
disp = ConfusionMatrixDisplay(cm_norm, display_labels=["Doublet", "Missing", "Singlet"])
disp.plot()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Prediction", fontsize=14)
plt.ylabel("Ground-truths", fontsize=14)
plt.tight_layout()
plt.savefig(f"map_out/for_expression/confusion_matrix_norm.png", dpi=150)
plt.close()