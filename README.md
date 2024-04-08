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
```bash
# it may take a while to finish setup
conda env create -f environment.yml
```

### Download image data, expression data
Please follow the READMEs in `data/` and `expression/`

### Download the pre-trained weights
Please follow the READMEs in `imagedoubler/model_data/` for the ResNet-50 weights and `imagedoubler/logs/` for ImageDoubler weights

## Train and evaluate model

To train from scratch, make sure the ResNet-50 pre-trained weight has been downloaded, and then:
```bash
cd imagedoubler/

# prepare the data splits for cross-validation
bash data_prepare.sh  

# integrated script for training all LOOCV and cross-resolution experiments, and evaluation, visualization
bash run.sh
```
You can also use our pre-trained model to skip the training, and directly run the commands for evaluation in `run.sh`

## Benchmark
Gene expression data can be downloaded following the README in `expression/`   
Environments can be set up by `conda env create -f scrna.yml`   
Scripts for running the other doublet detection methods are in `scripts/benchmark/`  
To compare and visualize the results, use `scripts/benchmark/benchmark.ipynb`


## Reference
[https://github.com/bubbliiiing/faster-rcnn-keras](https://github.com/bubbliiiing/faster-rcnn-keras)