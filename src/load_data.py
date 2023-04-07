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

        self.cache_dir = os.path.join(self.config['dataset_dir'] + '_cache', self.config['descriptor'])
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
            kpts = np.empty(0, 2, dtype=np.float32)
            desc = np.empty(0, 128, dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
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


def collate_function(batch, num_keypoints):
    """Pad/truncate and collate batched items"""
    M_list = []
    kpts0_list, kpts1_list = [], []
    desc0_list, desc1_list = [], []
    scores0_list, scores1_list = [], []
    mask0_list, mask1_list = [], []
    for data in batch:
        kpts0, desc0, scores0, mask0 = _pad_truncate_tensors(data['keypoints0'],
                                                             data['descriptors0'],
                                                             data['scores0'],
                                                             num_keypoints)
        kpts1, desc1, scores1, mask1 = _pad_truncate_tensors(data['keypoints1'],
                                                             data['descriptors1'],
                                                             data['scores1'],
                                                             num_keypoints)

        M_list.append(data['M'])
        kpts0_list.append(kpts0)
        kpts1_list.append(kpts1)
        desc0_list.append(desc0)
        desc1_list.append(desc1)
        scores0_list.append(scores0)
        scores1_list.append(scores1)
        mask0_list.append(mask0)
        mask1_list.append(mask1)

    batch = {
        'M': torch.stack(M_list),
        'keypoints0': torch.stack(kpts0_list),
        'keypoints1': torch.stack(kpts1_list),
        'descriptors0': torch.stack(desc0_list),
        'descriptors1': torch.stack(desc1_list),
        'scores0': torch.stack(scores0_list),
        'scores1': torch.stack(scores1_list),
        'mask0': torch.stack(mask0_list),
        'mask1': torch.stack(mask1_list)
    }
    return batch


def _pad_truncate_tensors(kpts, desc, scores, num_keypoints):
    """Pad/truncate input tensors to the lenght of num_keypoints"""
    mask = torch.ones(num_keypoints, dtype=torch.bool)

    if len(kpts) >= num_keypoints:
        kpts = kpts[:num_keypoints, :]
        desc = desc[:, :num_keypoints]
        scores = scores[:num_keypoints]
        return kpts, desc, scores, mask

    num_pad = num_keypoints - len(kpts)
    kpts = torch.concat((kpts, torch.zeros(num_pad, 2)))
    desc = torch.concat((desc, torch.zeros(num_pad, desc.size(1))))
    scores = torch.concat((scores, torch.zeros(num_pad)))
    mask[-num_pad:] = 0

    return kpts, desc, scores, mask


if __name__ == '__main__':
    from tqdm import tqdm
    from parse_args import parse_args

    args = parse_args()
    dataset = HomographyEstimationDataset(vars(args))
    for _ in tqdm(dataset, desc='Caching'):
        pass
