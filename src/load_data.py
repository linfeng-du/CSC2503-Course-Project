import os

import cv2
import torch
import numpy as np

from .utils import pad_tensors


class HomographyEstimationDataset(torch.utils.data.Dataset):
    default_config = {
        'dataset_dir': './dataset/revisitop1m',
        'resize_shape': [640, 480],
        'descriptor_type': 'SIFT',
        'num_keypoints': 1024
    }

    def __init__(self, config):
        super().__init__()
        config = {**self.default_config, **config}

        self.files = []
        for filename in os.listdir(config['dataset_dir']):
            self.files.append(os.path.join(config['dataset_dir'], filename))

        self.resize_shape = config['resize_shape']
        self.descriptor_type = config['descriptor_type']

        if self.descriptor_type == 'SIFT':
            self.num_keypoints = config['num_keypoints']
            self.sift = cv2.SIFT_create(nfeatures=self.num_keypoints)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        image, warped, M = self.get_image_pair(image)

        if self.descriptor_type != 'SIFT':
            return {
                'image0': torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
                'image1': torch.from_numpy(warped.astype(np.float32)).unsqueeze(0),
                'M': torch.from_numpy(M)
            }

        kpts0, desc0, scores0 = self.get_sift_tensors(image)
        kpts1, desc1, scores1 = self.get_sift_tensors(warped)

        kpts0, desc0, scores0, mask0 = pad_tensors(kpts0, desc0, scores0, self.num_keypoints)
        kpts1, desc1, scores1, mask1 = pad_tensors(kpts1, desc1, scores1, self.num_keypoints)

        return {
            'image0': torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            'image1': torch.from_numpy(warped.astype(np.float32)).unsqueeze(0),
            'M': torch.from_numpy(M),
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'mask0': mask0,
            'mask1': mask1
        }

    def get_image_pair(self, image):
        """Generate image pair via sampled homography"""
        h, w = image.shape
        corners = np.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp).astype(np.float32)
        warped = cv2.warpPerspective(image, M, (w, h))

        # Resize images
        image = cv2.resize(image, self.resize_shape)
        warped = cv2.resize(warped, self.resize_shape)

        return image, warped, M

    def get_sift_tensors(self, image):
        """Extract keypoints, descriptors, and confidence scores via SIFT"""
        kpts, desc = self.sift.detectAndCompute(image, None)
        num_kpts = min(self.num_keypoints, len(kpts))
        kpts_sift = kpts[:num_kpts]

        kpts = np.array([kpt.pt for kpt in kpts_sift], dtype=np.float32)
        desc = desc[:num_kpts, :] / 256.
        scores = np.array([kpt.response for kpt in kpts_sift], dtype=np.float32)

        kpts = torch.from_numpy(kpts)
        desc = torch.from_numpy(desc)
        scores = torch.from_numpy(scores)
        return kpts, desc, scores
