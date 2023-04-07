import os
import sys

import cv2
import torch
import numpy as np
from scipy.spatial import distance

sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.superpoint import SuperPoint


class HomographyEstimationDataset(torch.utils.data.Dataset):
    default_config = {
        'dataset_dir': './dataset/revisitop1m',
        'resize': [640, 480],
        'descriptor': 'SIFT',
        'num_keypoints': 1024,
        'device': 'cpu'
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.files = []
        for filename in os.listdir(self.config['dataset_dir']):
            self.files.append(os.path.join(self.config['dataset_dir'], filename))

        if self.config['descriptor'] == 'SIFT':
            self.sift = cv2.SIFT_create(nfeatures=self.config['num_keypoints'])
        elif self.config['descriptor'] == 'SuperPoint':
            self.superpoint = SuperPoint({'max_keypoints': self.config['num_keypoints']})
            self.superpoint.to(self.config['device'])

        cache_root = self.config['dataset_dir'] + '_cache'
        self.cache_dir = os.path.join(cache_root, self.config['descriptor'])
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, f'{idx}.pt')
        if os.path.exists(cache_path):
            # Cache hit, load directly
            return torch.load(cache_path)

        # Process data and cache
        image = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        warped, M = self.get_warped_image(image, np.random.default_rng(seed=idx))

        image = cv2.resize(image, self.config['resize'])
        warped = cv2.resize(warped, self.config['resize'])

        if self.config['descriptor'] == 'SIFT':
            kpts0, desc0, scores0 = self.get_sift_features(image)
            kpts1, desc1, scores1 = self.get_sift_features(warped)
        elif self.config['descriptor'] == 'SuperPoint':
            kpts0, desc0, scores0 = self.get_superpoint_features(image)
            kpts1, desc1, scores1 = self.get_superpoint_features(warped)

        matches, mismatches = self.get_matches_and_mismatches(kpts0, kpts1, M)

        data = {
            'M': torch.from_numpy(M),
            'keypoints0': torch.from_numpy(kpts0),
            'keypoints1': torch.from_numpy(kpts1),
            'descriptors0': torch.from_numpy(desc0),
            'descriptors1': torch.from_numpy(desc1),
            'scores0': torch.from_numpy(scores0),
            'scores1': torch.from_numpy(scores1),
            'matches': torch.from_numpy(matches),
            'mismatches': torch.from_numpy(mismatches)
        }

        torch.save(data, cache_path)
        return data

    def get_warped_image(self, image, rng):
        """Generate warped image via sampled homography"""
        h, w = image.shape
        corners = np.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)], dtype=np.float32)
        offsets = rng.integers(-224, 225, size=(4, 2)).astype(np.float32)  
   
        M = cv2.getPerspectiveTransform(corners, corners + offsets).astype(np.float32)
        warped = cv2.warpPerspective(image, M, (w, h))
        return warped, M

    def get_sift_features(self, image):
        """Extract keypoints, descriptors, and confidence scores via SIFT"""
        kpts_sift, desc = self.sift.detectAndCompute(image, None)

        if len(kpts_sift) == 0:
            kpts = np.empty((0, 2), dtype=np.float32)
            desc = np.empty((0, 128), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            return kpts, desc, scores

        kpts_sift = kpts_sift[:self.config['num_keypoints']]
        desc = desc[:self.config['num_keypoints'], :] / 256.

        kpts = np.array([kpt.pt for kpt in kpts_sift], dtype=np.float32)
        scores = np.array([kpt.response for kpt in kpts_sift], dtype=np.float32)

        return kpts, desc, scores

    def get_superpoint_features(self, image):
        """Extract keypoints, descriptors, and confidence scores via SuperPoint"""
        image = torch.from_numpy(image.astype(np.float32)).view(1, 1, *image.shape)
        image = image.to(self.config['device'])

        with torch.no_grad():
            tensors = self.superpoint({'image': image})

        # Batch size equals one
        kpts = tensors['keypoints'][0].cpu().numpy()
        desc = tensors['descriptors'][0].T.cpu().numpy()
        scores = tensors['scores'][0].cpu().numpy()
        return kpts, desc, scores

    def get_matches_and_mismatches(self, kpts0, kpts1, M):
        """Generate ground truth matches and mismatches"""
        if len(kpts0) == 0 or len(kpts1) == 0:
            matches = np.empty((0, 2), dtype=np.int64)
            mismatches = np.empty((0, 2), dtype=np.int64)
            return matches, mismatches

        # Warp kpts0 via homography
        kpts0 = cv2.perspectiveTransform(kpts0.reshape((1, -1, 2)), M).reshape((-1, 2))

        # Calculate pairwise distances
        dists = distance.cdist(kpts0, kpts1)
        min01 = np.argmin(dists, axis=1)
        min10 = np.argmin(dists, axis=0)
        min10_val = np.min(dists, axis=0)

        # Apply constraints to get ground truth matches and mismatches
        #   1. reprojection error is less than 3
        #   2. being the argmin of row (kpts0 -> kpts1) and column (kpts1 -> kpts0) at the same time
        match0_flt1 = min10[min10_val < 3]
        match0_flt2 = np.where(min10[min01] == np.arange(min01.shape[0]))[0]

        match0 = np.intersect1d(match0_flt1, match0_flt2)
        mismatch0 = np.setdiff1d(np.arange(kpts0.shape[0]), match0)

        match1 = min01[match0]
        mismatch1 = np.setdiff1d(np.arange(kpts1.shape[0]), match1)

        matches = np.stack((match0, match1), axis=1)
        mismatches0 = np.stack((mismatch0, -1 * np.ones_like(mismatch0)), axis=1)
        mismatches1 = np.stack((-1 * np.ones_like(mismatch1), mismatch1), axis=1)
        mismatches = np.concatenate((mismatches0, mismatches1))
        return matches, mismatches


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
        if key in ['batch_matches', 'batch_mismatches']:
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
    data[f'keypoints{ix}'] = torch.concat((kpts, torch.zeros(num_pad, 2)))
    data[f'descriptors{ix}'] = torch.concat((desc, torch.zeros(num_pad, desc.size(1))))
    data[f'scores{ix}'] = torch.concat((scores, torch.zeros(num_pad)))
    data[f'mask{ix}'][-num_pad:] = 0
    return data


def _filter_matches(matches, num_keypoints, ix):
    """Remove matches with truncated keypoints at index `ix`"""
    return matches[matches[:, ix] < num_keypoints, :]


if __name__ == '__main__':
    from tqdm import tqdm
    from parse_args import parse_args

    args = parse_args()
    dataset = HomographyEstimationDataset(vars(args))
    for _ in tqdm(dataset, desc='Caching'):
        pass
