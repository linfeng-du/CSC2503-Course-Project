import os
import sys
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.superglue import SuperGlue
from load_data import HomographyEstimationDataset
from utils import collate_fn, batch_to


def train(args):
    # Load dataset and split
    dataset = HomographyEstimationDataset(vars(args))
    generator = torch.Generator().manual_seed(args.data_seed)
    train, val, test = random_split(dataset, [len(dataset) - 2048, 1024, 1024], generator)

    collater = partial(collate_fn, num_keypoints=args.num_keypoints)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collater)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collater)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collater)

    # Build model, optimizer, and scheduler
    superglue = SuperGlue(vars(args))
    optimizer = optim.Adam(superglue.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    train_loss = []
    for epoch in range(args.num_epochs):
        train_loss += train_epoch(superglue, train_loader, optimizer, scheduler, epoch, args.device)
        evaluate(superglue, val_loader, args.device)


def train_epoch(superglue, train_loader, optimizer, scheduler, epoch, device):
    superglue.to(device)
    superglue.train()

    loss_list = []
    with tqdm(train_loader) as pbar:
        for data in pbar:
            data = batch_to(data, device)
            pred_matches = superglue(data)

            loss_matches = -pred_matches[data['batch_matches'],
                                        data['matches'][:, 0],
                                        data['matches'][:, 1]].sum()
            loss_mismatches = -pred_matches[data['batch_mismatches'],
                                            data['mismatches'][:, 0],
                                            data['mismatches'][:, 1]].sum()
            loss = loss_matches + loss_mismatches
            loss_list.append(loss.item())

            pbar.set_description(f'Epoch {epoch} | Loss {loss_list[-1]}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 1:
                # Exponential decay until iteration 900K [Sarlin et al. 2020]
                scheduler.step()

    return loss_list


def evaluate(superglue, eval_loader, device):
    superglue.to(device)
    superglue.eval()

    for data in tqdm(eval_loader):
        data = batch_to(data, device)
        pred = superglue.predict(data)
        print(pred)
