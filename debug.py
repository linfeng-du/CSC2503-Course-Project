from src.train import train_epoch
from src.parse_args import parse_args
from src.pipeline import load_dataset, build_model, build_optimizer


args = parse_args()
train_loader, val_loader, test_loader = load_dataset(args)
superpoint, superglue = build_model(args)
optimizer, scheduler = build_optimizer(superglue, args)
train_epoch(superpoint, superglue, train_loader, optimizer, scheduler, args)
