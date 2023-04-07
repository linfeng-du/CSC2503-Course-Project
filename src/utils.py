import torch


def collate_fn(batch, num_keypoints):
    """Pad or truncate tensors and collate them"""
    batched = {
        'M': [],
        'keypoints0':    [], 'keypoints1':       [],
        'descriptors0':  [], 'descriptors1':     [],
        'scores0':       [], 'scores1':          [],
        'mask0':         [], 'mask1':            [],
        'matches':       [], 'mismatches':       [],
        'batch_matches': [], 'batch_mismatches': []
    }
    for batch_ix, data in enumerate(batch):
        data = _pad_or_truncate_tensors(data, num_keypoints, ix=0)
        data = _pad_or_truncate_tensors(data, num_keypoints, ix=1)

        for key, val in data.items():
            batched[key].append(val)

        batched['batch_matches'].append(
            batch_ix * torch.ones_like(data['matches'][:, 0])
        )
        batched['batch_mismatches'].append(
            batch_ix * torch.ones_like(data['mismatches'][:, 0])
        )

    for key, val in batched.items():
        if key in ['matches', 'mismatches', 'batch_matches', 'batch_mismatches']:
            batched[key] = torch.cat(val)
        else:
            batched[key] = torch.stack(val)

    return batched


def _pad_or_truncate_tensors(data, num_keypoints, ix):
    """Pad or truncate tensors to the lenght of num_keypoints"""
    kpts, desc, scores = \
        data[f'keypoints{ix}'], data[f'descriptors{ix}'], data[f'scores{ix}']

    data[f'mask{ix}'] = torch.ones(num_keypoints, dtype=torch.bool)

    if len(kpts) >= num_keypoints:
        data[f'keypoints{ix}'] = kpts[:num_keypoints, :]
        data[f'descriptors{ix}'] = desc[:num_keypoints, :]
        data[f'scores{ix}'] = scores[:num_keypoints]
        data['matches'] = _filter_matches(data['matches'], num_keypoints, ix)
        data['mismatches'] = _filter_matches(data['mismatches'], num_keypoints, ix)
        return data

    num_pad = num_keypoints - len(kpts)
    data[f'keypoints{ix}'] = torch.concat((kpts, torch.zeros(num_pad, kpts.size(1))))
    data[f'descriptors{ix}'] = torch.concat((desc, torch.zeros(num_pad, desc.size(1))))
    data[f'scores{ix}'] = torch.concat((scores, torch.zeros(num_pad)))
    data[f'mask{ix}'][-num_pad:] = 0
    return data


def _filter_matches(matches, num_keypoints, ix):
    """Remove matches with truncated keypoints at index `ix`"""
    return matches[matches[:, ix] < num_keypoints, :]


def batch_to(batch, device):
    """Move device for dict-like batch"""
    for key, val in batch.items():
        batch[key] = val.to(device)

    return batch
