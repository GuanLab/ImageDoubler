import os
import sys
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

save_to = sys.argv[1]  # e.g.: loocv/Image7, for_expression/

annotation_mode     = 0
classes_path        = 'model_data/class.txt'

# trainval_percent    = 0.8
train_percent       = 0.8
all_set         = ["Image" + str(i) for i in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]]

def generate_train_val_test(model_id):
    if "loocv" in save_to:
        the_image_set = save_to.split("/")[-1]
        seed = int(the_image_set.replace("Image", ""))
        test_set = [the_image_set]
        
        random.seed(seed+int(model_id)+42)
        trainval_set    = [image_set for image_set in all_set if image_set not in test_set]
        random.shuffle(trainval_set)
        train_size      = int(len(trainval_set) * train_percent)
        train_set       = trainval_set[:train_size]
        val_set         = trainval_set[train_size:]
        
    elif "for_expression" in save_to:  # same as cross_resolution high2low 
        seed = 13
        test_set = ["Image5", "Image11"]
        
        random.seed(seed+int(model_id)+42)  
        trainval_set    = [image_set for image_set in all_set if image_set not in test_set]
        random.shuffle(trainval_set)
        train_size      = int(len(trainval_set) * train_percent)
        train_set       = trainval_set[:train_size]
        val_set         = trainval_set[train_size:]

    print(trainval_set)
    print(train_set)
    print(val_set)
    print(test_set)
    return trainval_set, train_set, val_set, test_set
    
print(all_set)


VOCdevkit_path  = 'VOCdevkit'

VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes, _      = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
if __name__ == "__main__":
    
    for model_id in [1, 2, 3, 4, 5]:
    
        print(f"Generate txt in ImageSets for model {model_id}.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, f'VOC2007/ImageSets/{save_to}')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        trainval_set, train_set, val_set, test_set = generate_train_val_test(model_id)
        trainval  = [xml for xml in total_xml if xml.split("_")[0] in trainval_set]
        train     = [xml for xml in total_xml if xml.split("_")[0] in train_set]
        val       = [xml for xml in total_xml if xml.split("_")[0] in val_set]
        test      = [xml for xml in total_xml if xml.split("_")[0] in test_set]
        
        print("train and val size", len(trainval))
        print("train size", len(train))
        ftrainval   = open(os.path.join(saveBasePath, f'trainval_{model_id}.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath, f'test_{model_id}.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath, f'train_{model_id}.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath, f'val_{model_id}.txt'), 'w')  
        
        for name in trainval:
            name = name[:-4]+"\n"
            ftrainval.write(name)
        for name in train:
            name = name[:-4]+"\n"
            ftrain.write(name)
        for name in val:
            name = name[:-4]+"\n"
            fval.write(name)
        for name in test:
            name = name[:-4]+"\n"
            ftest.write(name) 
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

        print(f"Generate 2007_train.txt and 2007_val.txt for training model {model_id}.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, f'VOC{year}/ImageSets/{save_to}/{image_set}_{model_id}.txt'), encoding='utf-8').read().strip().split()
            list_file = open(f'train_val_split/{save_to}/{year}_{image_set}_{model_id}.txt', 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        
        
        
