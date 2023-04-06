## Dataset
### Homography estimation - revisitop1m
Download the first 100K distractor images in [revisitop1m](https://github.com/filipradenovic/revisitop) via script or from [here](https://drive.google.com/file/d/133qEx930S3Z6bz3HD1Jl3gxSOEBNiy_5/view?usp=sharing). Unzip and place it to `./dataset/revisitop1m`.
```sh
python ./script/download_revisitop1m.py --num_images 100000
```

### Homography estimation - COCO2014
Download the training set of COCO2014. Unzip and place it to `./dataset/COCO_train2014`.
```sh
wget http://images.cocodataset.org/zips/train2014.zip
```