import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, random_split

from .models.superglue import SuperGlue
from .models.superpoint import SuperPoint
from .load_data import HomographyEstimationDataset


def load_dataset(args):
    if args.dataset == 'Oxford and Paris':
        args.dataset_dir = './dataset/revisitop1m'
    elif args.dataset == 'COCO':
        args.dataset_dir = './dataset/COCO_train2014'

    # Use the first 100K examples, size of test set following [Sarlin et al. 2020]
    dataset = Subset(HomographyEstimationDataset(vars(args)), range(100000))
    generator = torch.Generator().manual_seed(args.data_seed)
    train, val, test = random_split(dataset, [len(dataset) - 2048, 1024, 1024], generator)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader


def build_model(args):
    if args.descriptor == 'SIFT':
        args.descriptor_dim = 128
        superpoint = None
    elif args.descriptor == 'SuperPoint':
        args.descriptor_dim = 256
        args.max_keypoints = args.num_keypoints
        superpoint = SuperPoint(vars(args))

    superglue = SuperGlue(vars(args))
    return superpoint, superglue


def build_optimizer(superglue, args):
    optimizer = optim.Adam(superglue.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    return optimizer, scheduler
