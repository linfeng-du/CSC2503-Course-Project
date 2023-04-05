import torch
import numpy as np


def get_matching_matrix(kpts0, kpts1, mask0, mask1, M):
    """Generate ground truth matching matrix for batched keypoints"""
    device = kpts0.device
    b, n, _ = kpts0.size()

    # Warp src -> dst
    kpts0_src = torch.cat((kpts0, torch.ones(b, n, 1, device=device)), dim=-1)
    kpts0_dst = torch.bmm(M, kpts0_src.transpose(-2, -1)).transpose(-2, -1)
    kpts0_dst /= kpts0_dst[:, :, 2:3]
    kpts0 = kpts0_dst[:, :, :2]

    # Calculate pairwise distances
    dists = torch.cdist(kpts0, kpts1)

    # Constraint on true matches
    #   1. reprojection error less than 3
    #   2. alignment between src -> dst and dst -> src
    con_rep = dists < 3

    con_row = torch.zeros(b, n, n, dtype=torch.bool, device=device)
    con_row = torch.scatter(con_row, 2, torch.argmin(dists, dim=2).unsqueeze(2), 1)

    con_col = torch.zeros(b, n, n, dtype=torch.bool, device=device)
    con_col = torch.scatter(con_col, 2, torch.argmin(dists, dim=1).unsqueeze(2), 1)

    matches = con_rep & con_row & con_col
    print(matches.sum())

    # Add dustbins
    matches = torch.cat((matches, ~(torch.sum(matches, dim=1).unsqueeze(1).bool())), dim=1)
    matches = torch.cat((matches, ~(torch.sum(matches, dim=2).unsqueeze(2).bool())), dim=2)
    matches[:, -1, -1] = 0

    mms = []
    for dist in dists:
        min01_val, min01 = torch.min(dist, axis=1)
        min10_val, min10 = torch.min(dist, axis=0)
        match0_flt1 = min10[min10_val < 3]

        match0_flt2 = torch.where(min10[min01] == torch.arange(min01.shape[0], device=device))[0]

        # Get true matches and mismatches
        match0 = np.intersect1d(match0_flt1.cpu(), match0_flt2.cpu())
        mismatch0 = np.setdiff1d(torch.arange(kpts0.shape[0], device=device).cpu(), match0)

        match1 = min01[match0]
        mismatch1 = np.setdiff1d(torch.arange(kpts1.shape[0], device=device).cpu(), match1.cpu())

        # Build matching matrix
        matchesx = torch.zeros((n + 1, n + 1), dtype=torch.bool)
        matchesx[match0, match1] = 1
        # matchesx[mismatch0, n] = 1
        # matchesx[n, mismatch1] = 1

        mms.append(matchesx)

    matches2 = torch.stack(mms).cuda()
    print(matches2.sum())

    # return matches


def pad_batched_tensors(batch, num_keypoints, ix):
    """Pad batched result produced by SuperPoint"""
    assert len(batch['keypoints']) == len(batch['descriptors'])
    assert len(batch['keypoints']) == len(batch['scores'])

    kpts_list, desc_list, scores_list, mask_list = [], [], [], []
    for kpts, desc, scores in zip(batch['keypoints'], batch['descriptors'], batch['scores']):
        kpts, desc, scores, mask = pad_tensors(kpts, desc.T, scores, num_keypoints)
        kpts_list.append(kpts)
        desc_list.append(desc)
        scores_list.append(scores)
        mask_list.append(mask)

    batch = {
        f'keypoints{ix}': torch.stack(kpts_list),
        f'descriptors{ix}': torch.stack(desc_list),
        f'scores{ix}': torch.stack(scores_list),
        f'mask{ix}': torch.stack(mask_list)
    }
    return batch


def pad_tensors(kpts, desc, scores, num_keypoints):
    """Pad input tensors to the lenght of num_keypoints"""
    device = kpts.device

    mask = torch.ones(num_keypoints, dtype=torch.bool, device=device)
    if len(kpts) == num_keypoints:
        return kpts, desc, scores, mask

    num_pad = num_keypoints - len(kpts)
    kpts = torch.concat((kpts, torch.zeros(num_pad, 2, device=device)))
    desc = torch.concat((desc, torch.zeros(num_pad, desc.size(1), device=device)))
    scores = torch.concat((scores, torch.zeros(num_pad, device=device)))
    mask[-num_pad:] = 0

    return kpts, desc, scores, mask
