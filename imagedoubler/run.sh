#!/bin/bash

# parallel training
# skip Image6 if you use labeler-2's labels or weights
for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train.py loocv/Image1 $model &
    CUDA_VISIBLE_DEVICES=1 python train.py loocv/Image2 $model &
    CUDA_VISIBLE_DEVICES=2 python train.py loocv/Image3 $model &
    CUDA_VISIBLE_DEVICES=3 python train.py loocv/Image5 $model &
    CUDA_VISIBLE_DEVICES=5 python train.py loocv/Image6 $model &
    wait

    CUDA_VISIBLE_DEVICES=0 python train.py loocv/Image8 $model &
    CUDA_VISIBLE_DEVICES=1 python train.py loocv/Image9 $model &
    CUDA_VISIBLE_DEVICES=2 python train.py loocv/Image10 $model &
    CUDA_VISIBLE_DEVICES=3 python train.py loocv/Image11 $model &
    CUDA_VISIBLE_DEVICES=5 python train.py loocv/Image7 $model &
    wait
done

for model in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python train.py for_expression/ $model
done

# evaluate
# skip Image6 if you use labeler-2's labels or weights
for i in 1 2 3 5 6 7 8 9 10 11; do
    for model in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=0 python get_map.py loocv/Image$i $model
        CUDA_VISIBLE_DEVICES=0 python predict.py loocv/Image$i $model
    done 
done

for model in 1 2 3 4 5; do
    python get_map.py for_expression/ $model
    python predict.py for_expression/ $model
done

# within labeler
python get_accuracy.py
python get_confusion.py
