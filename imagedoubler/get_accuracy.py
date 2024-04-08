
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, balanced_accuracy_score, f1_score)


def count_cells(detection_result_file, thresh=0.7):
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
    counter = Counter(pred_classes)
    return counter.most_common(1)[0][0]


image_sets = ["Image" + str(i) for i in [1,2,3,5,6,7,8,9,10,11]]
N = len(image_sets)

for image_set in image_sets:
    targets = pd.read_csv("../data/crop_target/labeler1_targets.csv")
    targets["image_set"] = targets["image_id"].apply(lambda x: x.split("_")[0])
    targets = targets[(targets.image_set == image_set) & ~targets.difficult]

    performance = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds_class = []
        for idx, target in targets.iterrows():
            image_id = target["image_id"]
            
            pred_classes = []
            for model in ["model1", "model2", "model3", "model4", "model5"]:
                pred_n_cells, pred_class = count_cells(f"map_out/loocv/{image_set}/detection-results-{model}/{image_id}.txt", thresh)
                pred_classes.append(pred_class)
            preds_class.append(vote(pred_classes))
            
        targets["preds_class"] = preds_class

        # accuracy, balanced accuracy, f1_scores
        acc = accuracy_score(targets["class"], targets["preds_class"])
        balance_acc = balanced_accuracy_score(targets["class"], targets["preds_class"])
        weighted_avg_f1 = f1_score(targets["class"], targets["preds_class"], average="weighted")
        avg_f1 = f1_score(targets["class"], targets["preds_class"], average="macro")

        performance[thresh] = {"accuracy": acc,
                            "balanced_accuracy": balance_acc,
                            "weighted_f1": weighted_avg_f1,
                            "avg_f1": avg_f1}

    with open(f"map_out/loocv/{image_set}/performance.json", "w") as J:
        json.dump(performance, J, indent=4)
        
avg_scores = []
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    thresh = str(thresh)
    accs, balance_accs, weighted_avg_f1s, avg_f1s = 0, 0, 0, 0
    
    for image_set in image_sets:
        performances = json.load(open(f"map_out/loocv/{image_set}/performance.json"))
        accs += performances[thresh]["accuracy"]
        balance_accs += performances[thresh]["balanced_accuracy"]
        weighted_avg_f1s += performances[thresh]["weighted_f1"]
        avg_f1s += performances[thresh]["avg_f1"]
    
    avg_scores.append([thresh, accs / N, balance_accs / N, weighted_avg_f1s / N, avg_f1s / N])

pd.DataFrame(avg_scores).to_csv("map_out/loocv/avg_scores.tsv", sep="\t", index=False, header=False)

# for expression 
targets = pd.read_csv("../data/crop_target/labeler1_targets.csv")
targets["image_set"] = targets["image_id"].apply(lambda x: x.split("_")[0])
targets = targets[(targets.image_set.isin(["Image5", "Image11"])) & ~targets.difficult]

scores = []
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    preds_class = []
    for idx, target in targets.iterrows():
        image_id = target["image_id"]
        
        pred_classes = []
        for model in ["model1", "model2", "model3", "model4", "model5"]:
            pred_n_cells, pred_class = count_cells(f"map_out/for_expression/detection-results-{model}/{image_id}.txt", thresh)
            pred_classes.append(pred_class)
        preds_class.append(vote(pred_classes))
        
    targets["preds_class"] = preds_class

    # accuracy, balanced accuracy, f1_scores
    acc = accuracy_score(targets["class"], targets["preds_class"])
    balance_acc = balanced_accuracy_score(targets["class"], targets["preds_class"])
    weighted_avg_f1 = f1_score(targets["class"], targets["preds_class"], average="weighted")
    avg_f1 = f1_score(targets["class"], targets["preds_class"], average="macro")
    
    scores.append([thresh, acc, balance_acc, weighted_avg_f1, avg_f1])

pd.DataFrame(scores).to_csv("map_out/for_expression/performance.tsv", sep="\t", index=False, header=False)