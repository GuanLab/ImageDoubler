Data for paper ImageDoubler: Image-based Doublet Identification in Single-Cell Sequencing


The model pre-trained weights:

- labeler1_weights.zip: The zip file compressed the pre-trained ImageDoubler weights for cross-validations (in loocv/Image*/) and for validating on the image sets with expression data (in for_expression/). The 5 models for ensemble were also included.

- labeler2_weights.zip: It contained the same content as labeler1_weights.zip. The weights were trained on the labels from the second labeler 

- voc_weights_resnet.h5: This pre-trained weight is for training a detection model from scratch on a custom dataset.


The images and labels

- image_data.zip: contained the raw and processed images of the cells, as well as the labels from two labeler
    
    - raw/: 10 raw sequencing images from 10 experiments. The "rawImageIndex" provided the correspondences between experiments and the image set IDs.
    
    - blocks/: the cut blocks from the raw images. The images were named as Image[image set id]_[ROW]_[COLUMN].png
    
    - crop_image/: the blocks further cropped at the U-pipe regions. They were used for training.
    
    - crop_labels1/: the labels of the cropped images saved in XML format. They included the information of the cells' bounding boxes
    
    - crop_labels2/: same as crop_labels1. Labels came from the second labeler 
    
    - crop_target/: summarized the labels, counted the cells for each blocks, and converted to the "Missing", "Singlet", and "Doublet" classes


The expression data

- expression.zip: contained the processed expression matrices at gene level and transcript level. We used the gene level for benchmark

    - C1-SUM149-H1975: the expression data for image set 5. 

    - C1-SUM149-SUM190: the expression data for image set 11.

- C1-SUM149-H1975.tar.gz: the raw sequencing results in fastq format of image set 5

- C1-SUM149-SUM190.tar.gz: the raw sequencing results in fastq format of image set 11 
