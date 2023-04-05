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
