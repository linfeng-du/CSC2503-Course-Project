#  SuperGlue - Reimplementation
This repository contains training and evaluation code for Superglue model with homography pairs generated from COCO dataset. The code is adapted from the inference only [official implementation](https://github.com/magicleap/SuperGluePretrainedNetwork) released by MagicLeap

  

##  Requirements
- torch>=1.8.1
- torch_scatter>=2.06
- matplotlib>=3.1.3
- opencv-python==4.1.2.30
- numpy>=1.18.1
- PyYAML>=5.4.1
- wandb (If logging of checkpoints to cloud is need)
- albumentations 

##  Training data
COCO 2017 dataset is used for training. Random homographies are generated at every iteration and matches are computed using the know homography matrix. Download the 'train2017', 'val2017', and 'annotations' folder of COCO 2017 dataset and put that path in the config file used for training.

##  Training

All the parameters of training are provided in the coco_config.yaml in the configs folder. Change that file inplace and start the training. Or clone that file to make the changes and mention the custom config path in training command. Parameters of training are explained in comments in the coco_config.yaml file. To start the training run,

  

    python3 train_superglue.py --config_path configs/coco_config.yaml

Incase of Multi-GPU training, distributed setting is used. So run the following command,

  

    python3 -m torch.distributed.launch --nproc_per_node="NUM_GPUS" train_superglue.py --config_path configs/coco_config.yaml

Only singe-node training is supported as of now.

Checkpoints are saved at the end of every epoch, and best checkpoint is determined by weighted score of AUC at different thresholds, precision and recall computed on COCO 2017 val dataset using random homographies. Validation score is computed on fixed set of images and homographies for consistency across runs. Image and homography info used for validation is present at assets/coco_val_images_homo.txt

  
##  Evaluation

The official implementation has evaluation code for testing on small set of scannet scene pairs. Since our model in trained with random homographies, evaluating on scenes with random 3D camera movements doesn't perform well as pretrained indoor model. Instead we evaluate on test images of COCO, indoor and outdoor dataset(https://dimlrgbd.github.io/) with random homographies. Images are selected from the datasets and random homographies are generated for each of them. Based on matches given by the model, we determine the homography matrix using DLT and RANSAC implementation. As mentioned in paper, we report the AUC at 5, 10, 25 thresholds(for corner points), precision and recall. For evaluation run the following command,

  

    python3 match_homography.py --eval --superglue coco_homo

Parameter --superglue determines the checkpoint used and should be one of the following,

  

- Use **coco_homo** to run with the released coco homography model

- Use **PATH_TO_YOUR_.PT** to run with your trained model

- Use **indoor** to run with official indoor pretrained model

- Use **outdoor** to run with official outdoor pretrained model

  

Add --viz flag to dump the matching info image to 'dump_homo_pairs' folder.

If you want to evaluate with scannet pairs, run the above command with match_pairs.py with same parameters