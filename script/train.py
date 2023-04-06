import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.train import train
from src.parse_args import parse_args


if __name__ == '__main__':
    args = parse_args()
    train(args)
