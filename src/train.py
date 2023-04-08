import os
import sys
import json
from functools import partial

import torch
import torch.optim as optim
from torch_scatter import scatter_mean
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader, random_split
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.superglue import SuperGlue
from load_data import HomographyEstimationDataset
from metrics import homography_estimation
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

    # Build model (with seed), optimizer, and scheduler
    seed_everything(args.model_seed)
    superglue = SuperGlue(vars(args)).to(args.device)
    optimizer = optim.Adam(superglue.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    # Initialize tensorboard writer
    writer = SummaryWriter(os.path.join(args.logdir, 'tensorboard'))

    # Seed training before training
    seed_everything(args.model_seed)

    n_iter = 0
    best_val = 0.
    best_val_test = None
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader)
        for data in pbar:
            num_mat, num_mis = len(data['batch_matches']), len(data['batch_mismatches'])
            if num_mat == 0:
                continue

            data = batch_to(data, args.device)
            pred_matches = superglue(data)

            mis_ixs = torch.randperm(num_mis)[:num_mat]
            loss_matches = -scatter_mean(
                pred_matches[data['batch_matches'],
                             data['matches'][:, 0],
                             data['matches'][:, 1]],
                data['batch_matches']
            ).mean()
            loss_mismatches = -scatter_mean(
                pred_matches[data['batch_mismatches'][mis_ixs],
                             data['mismatches'][:, 0][mis_ixs],
                             data['mismatches'][:, 1][mis_ixs]],
                data['batch_mismatches'][mis_ixs]
            ).mean()
            loss = loss_matches + loss_mismatches

            writer.add_scalars('Loss/train', {
                'total': loss,
                'matches': loss_matches,
                'mismatches': loss_mismatches
                }, n_iter)
            pbar.set_description(f'Epoch {epoch} | Loss {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > 1:
                # Exponential decay until iteration 900K [Sarlin et al. 2020]
                scheduler.step()

            n_iter += 1
            if n_iter % 500 == 0:
                cur_val = evaluate(superglue, val_loader, args.device, args.resize, fit_homo=False)['F1']
                if cur_val > best_val:
                    best_val = cur_val
                    best_val_test = evaluate(superglue, test_loader, args.device, args.resize)
                    torch.save(superglue.state_dict(), os.path.join(args.logdir, 'model.pt'))
                    with open(os.path.join(args.logdir, 'test.json'), 'w') as fout:
                        json.dump(best_val_test, fout, indent=4)

                superglue.train()


def evaluate(superglue, eval_loader, device, resize, fit_homo=True):
    # superglue.eval()

    result = {
        'Precision': [], 'Recall': [], 'F1': [],
        'MPE-DLT': [], 'MPE-RANSAC': []
    }
    for data in tqdm(eval_loader):
        data = batch_to(data, device)
        pred = superglue.predict(data)
        batch_result = homography_estimation(data, pred, resize, fit_homo=fit_homo)

        del data, pred

        for key, val in batch_result.items():
            result[key] += val

    for key, val in batch_result.items():
        if len(val):
            result[key] = sum(val) / len(val)

    return result
