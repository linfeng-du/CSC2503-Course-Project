from tqdm import tqdm

from .utils import batch_to, pad_batched_tensors, get_matching_matrix


def train_epoch(superpoint, superglue, train_loader, optimizer, scheduler, args):
    superglue.to(args.device)
    if superpoint is not None:
        superpoint.to(args.device)

    superglue.train()
    for data in tqdm(train_loader):
        data = batch_to(data, args.device)

        if superpoint is not None:
            result0 = superpoint({'image': data['image0']})
            result1 = superpoint({'image': data['image1']})
            data.update(pad_batched_tensors(result0, args.num_keypoints, ix=0))
            data.update(pad_batched_tensors(result1, args.num_keypoints, ix=1))

        matches = get_matching_matrix(data['keypoints0'],
                                      data['keypoints1'],
                                      data['mask0'],
                                      data['mask1'],
                                      data['M'])
        pred_matches = superglue(data)

        print(matches.size(), pred_matches.size())
