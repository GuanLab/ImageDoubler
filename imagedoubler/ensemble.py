
import os
import pandas as pd
from glob import glob
from collections import Counter
import argparse


def agg_vote(x: pd.Series):
    class_order = {"Doublet": 0, "Singlet": 1, "Missing": 2}
    counter = Counter(x)
    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], class_order[item[0]]))
    return sorted_items[0][0]


parser = argparse.ArgumentParser(description='Arguments for making ensemble on the images with the trained models')
parser.add_argument('--model-ids', type=str, nargs="+", required=True, help='The IDs of models to ensemble. E.g. 1 2 3')
args = parser.parse_args()

preds = []
for model in args.model_ids:
    detection_results = glob(f"img_out/detection-results-model{model}/*.txt")

    for res_file in detection_results:
        image_id = os.path.splitext(os.path.basename(res_file))[0]
        
        with open(res_file, "r") as Res:
            cells = Res.readlines()
            cells = [record for record in cells if float(record.split()[1]) > 0.8]
            n_cells = len(cells)
        if n_cells == 0:
            image_class = "Missing"
        elif n_cells == 1:
            image_class = "Singlet"
        elif n_cells >= 2:
            image_class = "Doublet"
        else:
            raise ValueError("Invalid cell number")
        
        preds.append([image_id, image_class, model])
        
preds = pd.DataFrame(preds, columns=["image_id", "pred_image_class", "model_id"])
preds_agg = preds.groupby(["image_id"]).agg({"pred_image_class": agg_vote})
preds = preds.pivot(index="image_id", columns="model_id", values="pred_image_class")
preds["Ensemble"] = preds_agg["pred_image_class"]

preds.to_csv("output.csv")