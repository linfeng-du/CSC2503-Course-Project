import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # HomographyEstimationDataset
    parser.add_argument('--num_keypoints', type=int, default=1024)
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480])
    parser.add_argument('--descriptor', choices=['SIFT', 'SuperPoint'], default='SIFT')

    # SuperPoint
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)

    # SuperGlue
    parser.add_argument('--sinkhorn_iterations', type=int, default=100)
    parser.add_argument('--match_threshold', type=float, default=0.2)

    # Training
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)

    return parser.parse_args()
