import torch


def get_matching_matrix(kpts0, kpts1, mask0, mask1, M):
    """Generate ground truth matching matrix for batched keypoints"""
    device = kpts0.device
    b, n, _ = kpts0.size()

    # Warp from kpts0 to kpts1 with homography using homogeneous coordiantes
    kpts0_src = torch.cat((kpts0, torch.ones(b, n, 1, device=device)), dim=-1)
    kpts0_dst = torch.bmm(M, kpts0_src.transpose(-2, -1)).transpose(-2, -1)
    kpts0_dst /= kpts0_dst[:, :, 2:3]
    kpts0 = kpts0_dst[:, :, :2]

    # Calculate pairwise distances, fill masked positions with inf
    dists = torch.cdist(kpts0, kpts1)
    dists.masked_fill_(mask0.unsqueeze(2) == 0, torch.inf)
    dists.masked_fill_(mask1.unsqueeze(1) == 0, torch.inf)

    # Apply constraints to get true matches
    #   1. reprojection error is less than 3
    #   2. being the argmin of row (kpts0 -> kpts1) and column (kpts1 -> kpts0) at the same time
    #   3. not being at masked positions (in case no keypoints are being detected)
    con_rep = dists < 3

    con_row = torch.zeros(b, n, n, dtype=torch.bool, device=device)
    con_row.scatter_(2, dists.argmin(dim=2).unsqueeze(2), 1)

    con_col = torch.zeros(b, n, n, dtype=torch.bool, device=device)
    con_col.scatter_(1, dists.argmin(dim=1).unsqueeze(1), 1)

    # baddbmm_cuda only supports float type
    con_mask = torch.bmm(mask0.unsqueeze(2).float(), mask1.unsqueeze(1).float()).bool()

    matches = con_rep & con_row & con_col & con_mask

    # Unmatched keypoints are matched to dustbins (set dustbin -> dustbin to 0)
    dustbin_row = ~matches.sum(dim=2).bool()
    matches = torch.cat((matches, dustbin_row.unsqueeze(2)), dim=2)

    dustbin_col = ~matches.sum(dim=1).bool()
    matches = torch.cat((matches, dustbin_col.unsqueeze(1)), dim=1)

    matches[:, -1, -1] = 0
    return matches


def batch_to(batch, device):
    """Move device for dict-like batch"""
    for key, val in batch.items():
        batch[key] = val.to(device)

    return batch
