import os
import subprocess

# Get current directory
curdir = os.getcwd()

# Function to create directory if it doesn't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# loocv
set_ids = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
for set_id in set_ids:
    image_set = f"Image{set_id}"

    create_dir(os.path.join('logs', 'loocv', image_set))
    create_dir(os.path.join('train_val_split', 'loocv', image_set))
    create_dir(os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'loocv', image_set))
    create_dir(os.path.join('map_out', 'loocv', image_set))
    create_dir(os.path.join('img', 'loocv', image_set))
    create_dir(os.path.join('img_out', 'loocv', image_set))

# for expression & cross-resolution high2low
create_dir('logs/for_expression')
create_dir('train_val_split/for_expression')
create_dir('VOCdevkit/VOC2007/ImageSets/for_expression')
create_dir('map_out/for_expression')
create_dir('img/for_expression')
create_dir('img_out/for_expression')

# Change directory to img and run cp_data.py
os.chdir('img')
subprocess.run(['python', 'cp_data.py'])
os.chdir(curdir)

# Change directory to VOCdevkit/VOC2007 and run cp_data.py
os.chdir(os.path.join('VOCdevkit', 'VOC2007'))
subprocess.run(['python', 'cp_data.py'])
os.chdir(curdir)

# Run voc_annotation.py for each image set in loocv
for set_id in set_ids:
    image_set = f"Image{set_id}"
    subprocess.run(['python', 'voc_annotation.py', f'loocv/{image_set}'])

# Run voc_annotation.py for for_expression
subprocess.run(['python', 'voc_annotation.py', 'for_expression'])

