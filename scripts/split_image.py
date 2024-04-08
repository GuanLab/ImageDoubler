#!/usr/bin/env python

## This code is used to split the raw image into 40x20 small images,
## and named them in the ImageX_rowNum_colNum.png format.
## input:name of raw image, index name of image, h1, w1, h2, w2. 
##       Index are stored in the file ../rawImageIndex
## output: splitted images
## useage: python preprocess.py "Chip A Sample A (Left) Sample B (Right).jpg" "Image1" 0 0 0 0

def cut_image(name,image_index,height_cut_head,width_cut_head,height_cut_tail,width_cut_tail):
  import cv2
  import numpy as np
  import math
  import sys

  path= "/local/disk3/mqzhou/single_cell_image/raw/"

  save_path='/local/disk3/mqzhou/single_cell_image/split_image/'+image_index+'/'
  img=cv2.imread(path+name)
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  nrow,ncol=gray.shape#height,width
  gray=gray[int(height_cut_head):(nrow-int(height_cut_tail)),int(width_cut_head):(ncol-int(width_cut_tail))]
  nrow,ncol=gray.shape
  row_unit=nrow//40
  col_unit=ncol//20
  for i in range(40):#40 row
      for j in range(20):#20 col
          row_start=max(math.ceil(i*row_unit),0)
          row_end=max(math.ceil((i+1)*row_unit),0)
          col_start=max(math.ceil(j*col_unit),0)
          col_end=max(math.ceil((j+1)*col_unit),0)
          subimg=gray[int(row_start):int(row_end),int(col_start):int(col_end)]
          cv2.imwrite(save_path+str(image_index)+'_'+str(i+1)+'_'+str(j+1)+'.png',subimg)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
    description='Split the raw image into subimages')
  parser.add_argument('--image', required=True,
    metavar="string",
    help='name of the raw image')
  parser.add_argument('--save', required=True,
    metavar="string",
    help="save name of the raw image")
  parser.add_argument('--h1', required=False,
    default=0,
    metavar="integer",
    help='head cut length')
  parser.add_argument('--w1', required=False,
    default=0,
    metavar="integer",
    help='left side cut length')
  parser.add_argument('--h2', required=False,
    default=0,
    metavar="integer",
    help='tail cut length')
  parser.add_argument('--w2', required=False,
    default=0,
    metavar="integer",
    help='right side cut length')
  args = parser.parse_args()

  if args.h1:
    print('The raw image head will be cut with length ',args.h1)
  if args.w1:
    print('The raw image left side will be cut with length',args.w1)
  if args.h2:
    print('The raw image tail will be cut with length ',args.h2)
  if args.w2:
    print('The raw image right side will be cut with length',args.w2)

  cut_image(args.image,args.save,args.h1,args.w1,args.h2,args.w2)


#python preprocess.py --image="Sh_SUM149 (left) and SUM190 (right)_HT chip (4-19-18).jpg" --save="Image11" --h1=100 --w1=300 --h2=100 --w2=300
#python preprocess.py --image="Sh_C1 HT_SUM149 (left) and H1975 (right) 8-7-18.jpg" --save="Image5" --h1=50 --w1=300 --h2=250 --w2=300
#python preprocess.py --image="Sh_Noriaki_G+ and G+T+samples C1 HT 800 (3-14-18).jpg" --save="Image9" --h1=300 --w1=100 --h2=400 --w2=0
#python preprocess.py --image="Sh_Sammi Wt (Left 1-10) and KO (right 11-20) HT 800 RNAseq (1-31-18).jpg" --save="Image10" --h1=350 --w1=100 --h2=350 --w2=0

