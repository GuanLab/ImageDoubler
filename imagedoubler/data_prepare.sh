#!/bin/bash

curdir=$(pwd)

cd ./img
python cp_data.py
cd $curdir

cd ./VOCdevkit/VOC2007
python cp_data.py
cd $curdir

# loocv
for set_id in 1 2 3 5 6 7 8 9 10 11; do
    image_set=Image"$set_id"

    if [ ! -d logs/loocv/"$image_set" ]; then
        mkdir -p logs/loocv/"$image_set"
    fi

    if [ ! -d train_val_split/loocv/"$image_set" ]; then
        mkdir -p train_val_split/loocv/"$image_set"
    fi

    if [ ! -d VOCdevkit/VOC2007/ImageSets/loocv/"$image_set" ]; then
        mkdir -p VOCdevkit/VOC2007/ImageSets/loocv/"$image_set"
    fi

    if [ ! -d map_out/loocv/"$image_set" ]; then
	    mkdir -p map_out/loocv/"$image_set"
    fi

    if [ ! -d img/loocv/"$image_set" ]; then
	    mkdir -p img/loocv/"$image_set"
    fi

    if [ ! -d img_out/loocv/"$image_set" ]; then
	    mkdir -p img_out/loocv/"$image_set"
    fi

    python voc_annotation.py loocv/"$image_set"

done


# for expression & cross-resolution high2low
if [ ! -d logs/for_expression ]; then
    mkdir -p logs/for_expression
fi

if [ ! -d train_val_split/for_expression ]; then
    mkdir -p train_val_split/for_expression
fi

if [ ! -d VOCdevkit/VOC2007/ImageSets/for_expression ]; then
    mkdir -p VOCdevkit/VOC2007/ImageSets/for_expression
fi

if [ ! -d map_out/for_expression ]; then
    mkdir -p map_out/for_expression
fi

if [ ! -d img/for_expression ]; then
    mkdir -p img/for_expression
fi

if [ ! -d img_out/for_expression ]; then
    mkdir -p img_out/for_expression
fi

python voc_annotation.py for_expression/

cd train_val_split/
python check_overlaps.py
cd $curdir