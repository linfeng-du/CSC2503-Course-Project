## Requirements
- torch==1.13.0
- torch-scatter==2.1.1+pt113cu117
- opencv-python==4.7.0.72
- numpy==1.23.5
- matplotlib==3.6.2


## Dataset
### Oxford and Paris
Download the first 100K distractor images in [revisitop1m](https://github.com/filipradenovic/revisitop).
```sh
python ./script/download_revisitop1m.py --num_images 100000
```

### COCO2014
Download the training set of COCO2014 (82783 images). Unzip and place it to `./dataset/COCO_train2014`.
```sh
wget http://images.cocodataset.org/zips/train2014.zip
```


## Extract keypoint features and cache
Extract keypoint features as a pre-processing step to alleviate the dataloader bottleneck. This step will also randomly generate a warped image for each original image and extract keypoint features for the warped image.
```sh
python ./src/load_data.py --dataset "Oxford and Paris" --descriptor SIFT --num_keypoints 1024 --device cpu
python ./src/load_data.py --dataset "Oxford and Paris" --descriptor SuperPoint --num_keypoints 512 --device cuda

python ./src/load_data.py --dataset COCO --descriptor SIFT --num_keypoints 1024 --device cpu
python ./src/load_data.py --dataset COCO --descriptor SuperPoint --num_keypoints 512 --device cuda
```

## Training
### Training with SIFT descriptors
```sh
python ./script/train.py --dataset "Oxford and Paris" --descriptor SIFT --num_keypoints 1024
python ./script/train.py --dataset COCO --descriptor SIFT --num_keypoints 1024
```

### Training with SuperPoint descriptors
```sh
python ./script/train.py --dataset "Oxford and Paris" --descriptor SuperPoint --num_keypoints 512
python ./script/train.py --dataset COCO --descriptor SuperPoint --num_keypoints 512
```


## Running on CSLab Slurm cluster
```sh
sinfo   # Check gpunode availability and partition info
srun --partition ${partition} --nodelist ${gpunode} --gres gpu:1 --cpus-per-task 4 --mem 8G --pty bash
```
