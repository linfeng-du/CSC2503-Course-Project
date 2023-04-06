from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # HomographyEstimationDataset
    parser.add_argument('--dataset', choices=['Oxford and Paris', 'COCO'], default='Oxford and Paris')
    parser.add_argument('--num_keypoints', type=int, default=1024)
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480])
    parser.add_argument('--descriptor', choices=['SIFT', 'SuperPoint'], default='SIFT')

    # SuperPoint
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--remove_borders', type=int, default=4)

    # SuperGlue
    parser.add_argument('--weights', choices=['none', 'indoor', 'outdoor'], default='none')
    parser.add_argument('--sinkhorn_iterations', type=int, default=100)
    parser.add_argument('--match_threshold', type=float, default=0.2)

    # Training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.999998)

    # Reproducibility
    parser.add_argument('--homo_seed', type=int, default=42)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--model_seed', type=int, default=42)
    parser.add_argument('--training_seed', type=int, default=42)

    args = parser.parse_args()

    if args.dataset == 'Oxford and Paris':
        args.dataset_dir = './dataset/revisitop1m'
    elif args.dataset == 'COCO':
        args.dataset_dir = './dataset/COCO_train2014'

    if args.descriptor == 'SIFT':
        args.descriptor_dim = 128
    elif args.descriptor == 'SuperPoint':
        args.descriptor_dim = 256

    return parser.parse_args()
