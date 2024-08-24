# **ImageDoubler: Image-based Doublet Identification in Single-Cell Sequencing**

This is the package of ImageDoubler, an image-based doublet detection model implemented on Faster-RCNN. It is trained on the images from the Fluidigm C1 platform. Please contact ([dengkw@umich.edu](mailto:dengkw@umich.edu) or [gyuanfan@umich.edu](mailto:gyuanfan@umich.edu)) if you have any questions or suggestions.

![Figure1](fig/Figure1.png?raw=true "Title")

---

## Installations

### Git clone this repository:

```bash
git clone https://github.com/GuanLab/ImageDoubler.git
```

### Setup the running environment

- For training/evaluating the model:
```bash
# it may take about 10 - 15 minutes to finish setup
conda env create -f environment.yml
```

- For benchmark:
```bash
# it may take about 5 minutes to finish setup
conda env create -f scrna.yml

# for running the SoCube, which may need python 3.9 for its installation
conda create -n socube python=3.9
pip install socube
```

Creating environments from the .yml files sometimes may fail. Users can also try the conda commands in `setup_environment.sh` for setting the environments

----

## Repeat the training and evaluation results in the paper

### Download image data, processed expression data
Please follow the READMEs in `data/` and `expression/`

### Download the pre-trained weights
Please follow the READMEs in `imagedoubler_paper/model_data/` for the ResNet-50 weights and `imagedoubler_paper/logs/` for ImageDoubler weights

### Train and evaluate from scratch
The following codes have been tested on both Linux and Windows system. Make sure that you have downloaded the ResNet-50 weights has been downloaded.

1. Prepare the data splits for cross-validation
```bash
cd imagedoubler_paper/
python data_prepare.py 
```

2. Train models for LOOCV
```bash
# python train.py loocv/[Image_set_for_test] [model_num]
# Here showed an example of using the images from image set 1 as the test set 
# The other images are used as training and validation
for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train.py loocv/Image1 $model
done
```

3. Train models for evaluations with expression data
```bash
# Images from image sets 5 and 11 are used as test data
for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train.py for_expression/ $model
done
```

4. Generate inferences for LOOCV
```bash
for model in 1 2 3 4 5; do
    # retrieve the cells' confidence score and position
    # e.g.: cell 0.9895 48 73 55 79
    CUDA_VISIBLE_DEVICES=0 python get_map.py loocv/Image1 $model
    # generate the images with bounding boxes  
    CUDA_VISIBLE_DEVICES=0 python predict.py loocv/Image1 $model
done
```

5. Generate inferences for evaluation with expression data
```bash
for model in 1 2 3 4 5; do
    python get_map.py for_expression/ $model  
    python predict.py for_expression/ $model
done
```

6. Evaluations
```bash
python get_accuracy.py
python get_confusion.py
```

The full pipeline is integrated in `run.sh`. Linux users can directly run it by `bash run.sh`. 

You can also use our pre-trained model to skip the training (steps 1-3), and directly run the inference or evaluation codes.

### Process the sequencing data
The processed expression matrix is provided. It can be downloaded from [expression.zip](https://www.dropbox.com/scl/fo/z75nudrjp2e2nqqy954zo/h?e=2&preview=expression.zip&rlkey=e5oi5vbpl559uynmj10yi3xi5&st=x9tziyis&dl=0). Move it and unzip under the `expression/` folder.

To start from the raw data, they can be downloaded at:
- Image set 5: [C1-SUM149-H1975](https://www.dropbox.com/scl/fo/z75nudrjp2e2nqqy954zo/h?e=2&preview=C1-SUM149-H1975.tar.gz&rlkey=e5oi5vbpl559uynmj10yi3xi5&st=na3d6dg5&dl=0)
- Image set 11: [C1-SUM149-SUM190](https://www.dropbox.com/scl/fo/z75nudrjp2e2nqqy954zo/h?e=2&preview=C1-SUM149-SUM190.tar.gz&rlkey=e5oi5vbpl559uynmj10yi3xi5&st=ih8g3zmb&dl=0)

Codes for processing are available at `scripts/expression/`, you may need to modify the data paths in the codes accordingly:

1. Demultiplex
```bash
./mRNASeqHT_demultiplex.pl -i input/dir/of/fastq/data/ -o output/dir/
# outputs may include the files like:
#   - 114709_TAAGGCGA_S1_ROW01_R1.fastq
#   - 114709_TAAGGCGA_S1_ROW01_R2.fastq
#   - ...
#   - 114709_TAAGGCGA_S1_ROW40_R2.fastq
#   - 114709_TAAGGCGA_S1-Undetermined_R1.fastq
#   - 114709_TAAGGCGA_S1-Undetermined_R2.fastq
```
2. Extract expression data with kallisto
```bash
# C1-SUM149-H1975_columns.csv for image set 5
# C1-SUM149-SUM190_columns.csv fo image set 11
python run_kallisto.py the_column_file.csv
```
3. Arrange the expression data into a matrix with tximport
```bash
Rscript generate_count_matrix.R
```

### Benchmark  

Scripts for running the other doublet detection methods are in `scripts/benchmark/`  
To compare and visualize the results, use `scripts/benchmark/benchmark.ipynb`
The data/file paths in these scripts need to be adjusted accordingly. 

--------------------------

## Train and inference with custom image data

### Download the pre-trained weights
- (If you train from scratch) Download the ResNet-50 weights in `imagedoubler/model_data/`
- (If you finetune on ImageDoubler or use it just for inference) Download the ImageDoubler's weights in `imagedoubler/logs/`

### Prepare the inputs for training
1. Images of the cells
2. The annotation file include the information of training images, which should contain the information like the following example:
```
# columns are:
# path_of_image xmin1,ymin1,xmax1,ymax1,0 xmin2,ymin2,xmax2,ymax2,0 ...
JPEGImages/Image11_40_12.jpg 46,70,52,77,0
JPEGImages/Image7_16_13.jpg
JPEGImages/Image8_12_20.jpg 118,181,134,192,0 211,143,219,156,0
JPEGImages/Image2_15_7.jpg 120,180,129,189,0 188,41,194,50,0
JPEGImages/Image7_9_12.jpg 118,179,130,190,0
...
```
3. The annotation file include the information of validation images. It contains similar information as the train annotation organized in the same format

### Training
```bash
cd imagedoubler
python train.py --train-anno path/to/annotation_images_train.txt \
                --val-anno path/to/annotation_images_validation.txt \
                --model-id 0 \  # a specific ID to name the model
                --pretrain-weight path/to/resnet_weight_or_ImageDoubler_weight.h5 \
                --out-dir path/to/directory_of_saving_weights_and_logs/
```

### Making inference
Note: User can skip the training phase and use the ImageDoubler's weights to infer their custom images.

```bash
python predict.py --model-path path/to/custom_trained_or_ImageDoubler_model_weight.h5 \
                  --model-id 0 \  # should be consistent with the ID in the model name
                  --conf 0.7 \  # The confidence threshold for the detection
                  --image-dir path/to/directory_of_test_images/ \
                  --out-dir path/to/directory_for_save_inference_results/
```
The test images with inferred bounding boxes can be checked at: 
`<out_dir>/detection-results-img-model<model_id>`

### Ensemble and generate final decision
```bash
# List the IDs for the models that you want to ensemble after --model-ids, 
# IDs should be separated by space
# IDs should be those given to train.py and predict.py
python ensemble.py --model-ids 1 2 3 4 5
```

This program will generate a `output.csv` file with the contents like below, indicating the cell condition for each image. 
```
image_id,1,2,3,4,5,Ensemble
Image1_10_1,Singlet,Singlet,Singlet,Singlet,Singlet,Singlet
Image1_10_10,Singlet,Doublet,Doublet,Doublet,Doublet,Doublet
Image1_10_11,Singlet,Doublet,Singlet,Singlet,Doublet,Singlet
Image1_10_12,Singlet,Singlet,Singlet,Singlet,Singlet,Singlet
Image1_10_13,Singlet,Singlet,Singlet,Singlet,Singlet,Singlet
...
```

--------------------------

## Reference
The Faster-RCNN implementation is based on the codes from: [https://github.com/bubbliiiing/faster-rcnn-keras](https://github.com/bubbliiiing/faster-rcnn-keras)