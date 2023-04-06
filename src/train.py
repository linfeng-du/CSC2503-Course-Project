import os
import sys
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.superglue import SuperGlue
from load_data import HomographyEstimationDataset, collate_function
from utils import batch_to, pad_batched_tensors, get_matching_matrix


def train(args):
    # Load dataset and split
    dataset = HomographyEstimationDataset(vars(args))
    generator = torch.Generator().manual_seed(args.data_seed)
    train, val, test = random_split(dataset, [len(dataset) - 2048, 1024, 1024], generator)

    collate_fn = partial(collate_function, num_keypoints=args.num_keypoints)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_fn)

    # Build model, optimizer, and scheduler
    superglue = SuperGlue(vars(args))
    optimizer = optim.Adam(superglue.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        train_epoch(superglue, train_loader, optimizer, scheduler, args.device)


def train_epoch(superglue, train_loader, optimizer, scheduler, device):
    superglue.to(device)

    superglue.train()
    for data in tqdm(train_loader):
        data = batch_to(data, device)

        matches = get_matching_matrix(data['keypoints0'],
                                      data['keypoints1'],
                                      data['mask0'],
                                      data['mask1'],
                                      data['M'])
        pred_matches = superglue(data)

        loss = -(pred_matches * matches).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(matches.size(), pred_matches.size())
