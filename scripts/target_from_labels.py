
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from glob import glob


def label_files_to_target_xml(label_files):
    targets = []
    for label_file in label_files:
        image_id = os.path.splitext(os.path.basename(label_file))[0]
        contents = open(label_file, "r").read()
        soup = BeautifulSoup(contents, "html.parser")
        
        objects = soup.find_all("object")
        num_of_cell = len(objects)
        is_difficult = any([int(o.find("difficult").string) for o in objects])
        if num_of_cell == 0:
            the_class = "Missing"
        elif num_of_cell == 1:
            the_class = "Singlet"
        elif num_of_cell >= 2:
            the_class = "Doublet"
        else:
            raise ValueError("Number of cell should not be a negative number")
        
        targets.append([image_id, num_of_cell, the_class, is_difficult])
    targets = pd.DataFrame(targets, columns=["image_id", "cell_num", "class", "difficult"])
    return targets


def label_files_to_target_csv(label_files):

    def num_to_class(cell_num):
        if cell_num == 0:
            return "Missing"
        elif cell_num == 1:
            return "Singlet"
        elif cell_num >= 2:
            return "Doublet"
        else:
            raise ValueError("Number of cell should not be a negative number")

    targets = pd.DataFrame()
    for label_file in label_files:
        if "image4" in label_file:
            continue
        target = pd.read_csv(label_file)
        target.columns = ["image_id", "cell_num"]
        target["image_id"] = target["image_id"].str.capitalize()
        target["class"] = target["cell_num"].apply(num_to_class)
        target["difficult"] = False
        targets = pd.concat([targets, target], ignore_index=True)

    return targets

    

label_files = glob("../data/crop_labels1/Image*/*.xml")
targets = label_files_to_target_xml(label_files)
targets.to_csv("../data/crop_target/labeler1_targets.csv", index=False)

label_files = glob("../data/crop_labels2/Image*/*.xml")
targets = label_files_to_target_xml(label_files)
targets.to_csv("../data/crop_target/labeler2_targets.csv", index=False)

    
    


