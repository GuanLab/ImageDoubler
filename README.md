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
```

### Download image data, processed expression data
Please follow the READMEs in `data/` and `expression/`

### Download the pre-trained weights
Please follow the READMEs in `imagedoubler/model_data/` for the ResNet-50 weights and `imagedoubler/logs/` for ImageDoubler weights

## Train and evaluate model

To train from scratch, make sure the ResNet-50 pre-trained weight has been downloaded:
- Prepare the data splits for cross-validation
```bash
cd imagedoubler/
bash data_prepare.sh  
```
- Train models for LOOCV
```bash
# python train.py loocv/[Image_set_for_test] [model_num]
for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train.py loocv/Image1 $model
done
```
- Train models for evaluations with expression data
```bash
for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train.py for_expression/ $model
done
```
- Generate inferences for LOOCV
```bash
for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python get_map.py loocv/Image1 $model
    CUDA_VISIBLE_DEVICES=0 python predict.py loocv/Image1 $model
done
```
- Generate inferences for evaluation with expression data
```bash
for model in 1 2 3 4 5; do
    python get_map.py for_expression/ $model
    python predict.py for_expression/ $model
done
```
- Evaluations
```bash
python get_accuracy.py
python get_confusion.py
```

To run the integrated full pipeline, you can directly run the `run.sh` after `data_prepare.sh`.

You can also use our pre-trained model to skip the training, and directly run the inference or evaluation codes.

## Process the sequencing data

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

## Benchmark  
Environments can be set up by 
```bash
conda env create -f scrna.yml
```  
Scripts for running the other doublet detection methods are in `scripts/benchmark/`  
To compare and visualize the results, use `scripts/benchmark/benchmark.ipynb`

## Reference
[https://github.com/bubbliiiing/faster-rcnn-keras](https://github.com/bubbliiiing/faster-rcnn-keras)