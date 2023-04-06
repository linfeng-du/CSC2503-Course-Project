import os
import sys

import cv2
import torch
import numpy as np

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
            kpts0, desc0, scores0 = self.get_sift_tensors(image)
            kpts1, desc1, scores1 = self.get_sift_tensors(warped)
        elif self.config['descriptor'] == 'SuperPoint':
            kpts0, desc0, scores0 = self.get_superpoint_tensors(image)
            kpts1, desc1, scores1 = self.get_superpoint_tensors(warped)

        data = {
            'M': torch.from_numpy(M),
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1
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

    def get_sift_tensors(self, image):
        """Extract keypoints, descriptors, and confidence scores via SIFT"""
        kpts, desc = self.sift.detectAndCompute(image, None)

        if len(kpts) == 0:
            kpts = torch.empty(0, 2)
            desc = torch.empty(0, 128)
            scores = torch.empty(0)
            return kpts, desc, scores

        num_kpts = min(self.config['num_keypoints'], len(kpts))
        kpts_sift = kpts[:num_kpts]

        kpts = np.array([kpt.pt for kpt in kpts_sift], dtype=np.float32)
        desc = desc[:num_kpts, :] / 256.
        scores = np.array([kpt.response for kpt in kpts_sift], dtype=np.float32)

        kpts = torch.from_numpy(kpts)
        desc = torch.from_numpy(desc)
        scores = torch.from_numpy(scores)
        return kpts, desc, scores

    def get_superpoint_tensors(self, image):
        """Extract keypoints, descriptors, and confidence scores via SuperPoint"""
        image = torch.from_numpy(image.astype(np.float32)).view(1, 1, *image.shape)
        image = image.to(self.config['device'])
        tensors = self.superpoint({'image': image})

        kpts = tensors['keypoints'][0].to('cpu')
        desc = tensors['descriptors'][0].T.to('cpu')
        scores = tensors['scores'][0].to('cpu')
        return kpts, desc, scores


if __name__ == '__main__':
    from tqdm import tqdm
    from parse_args import parse_args

    args = parse_args()
    dataset = HomographyEstimationDataset(vars(args))
    for _ in tqdm(dataset, desc='Caching'):
        pass
