import torch
from tqdm import tqdm

from src.train import train_epoch
from src.parse_args import parse_args
from src.pipeline import load_dataset, build_model, build_optimizer
from src.models.superpoint import SuperPoint
from src.load_data import HomographyEstimationDataset


args = parse_args()
train_loader, val_loader, test_loader = load_dataset(args)
superpoint, superglue = build_model(args)
optimizer, scheduler = build_optimizer(superglue, args)
train_epoch(superpoint, superglue, train_loader, optimizer, scheduler, args)





max_num = 0
dataset = HomographyEstimationDataset({'descriptor_type': 'SIFT'})
loader = torch.utils.data.DataLoader(dataset, batch_size=8)
for data in tqdm(loader):
    # image = data['image0'].cuda()
    # result = superpoint({'image': image})
    # batch = pad_batched_tensors(result, 512, ix=0)
    # kpts0 = batch['keypoints0']

    # image = data['image1'].cuda()
    # result = superpoint({'image': image})
    # batch.update(pad_batched_tensors(result, 512, ix=1))
    # kpts1 = batch['keypoints1']

    print(data['descriptors0'].size())
    kpts0 = data['keypoints0'].cuda()
    kpts1 = data['keypoints1'].cuda()
    mask0 = data['mask0'].cuda()
    mask1 = data['mask1'].cuda()
    M = data['M'].cuda()
    get_matching_matrix(kpts0, kpts1, mask0, mask1, M)
