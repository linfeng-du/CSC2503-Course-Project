## Dataset
### Homography estimation - Oxford and Paris
Download the first 100K distractor images in [revisitop1m](https://github.com/filipradenovic/revisitop). Unzip and place it to `./dataset/revisitop1m`.
```sh
python ./script/download_revisitop1m.py --num_images 100000
```

### Homography estimation - COCO2014
Download the training set of COCO2014 (82783 images). Unzip and place it to `./dataset/COCO_train2014`.
```sh
wget http://images.cocodataset.org/zips/train2014.zip
```

### Extract features and cache
Extract features as a pre-processing step to alleviate dataloader bottleneck.
```sh
python ./src/load_data.py --dataset "Oxford and Paris" --descriptor SIFT --num_keypoints 1024 --device cpu
python ./src/load_data.py --dataset "Oxford and Paris" --descriptor SuperPoint --num_keypoints 512 --device cuda

python ./src/load_data.py --dataset "COCO" --descriptor SIFT --num_keypoints 1024 --device cpu
python ./src/load_data.py --dataset "COCO" --descriptor SuperPoint --num_keypoints 512 --device cuda
```

## Running on CSLab Slurm cluster
```sh
sinfo   # Check gpunode availability and partition info
srun --partition ${partition} --nodelist ${gpunode} --cpus-per-task 4 --gres=gpu:1 --mem=8G --pty bash
```
