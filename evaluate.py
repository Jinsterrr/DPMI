#!/usr/bin/env python3
"""
train.py - Unified training launcher for all main algorithms in this repository.

Example usage:
    python train.py --algo adp_clip --net resnet --dataset cifar10 --data_root ./data --batchsize 256 --epoch 60 --private --eps 8.0 --delta 1e-5

Supported algorithms:
    - adp_clip
    - adp_alloc
    - dp_sgd
    - loss

This script will call the corresponding main.py script in the appropriate subfolder with the provided arguments.
"""
import argparse
import subprocess
import sys
import os

ALGO_PATHS = {
    'adp_clip': 'dp_mechanism/DP_training/DPMLBench/algorithms/adp_clip/main.py',
    'adp_alloc': 'dp_mechanism/DP_training/DPMLBench/algorithms/adp_alloc/main.py',
    'dp_sgd': 'dp_mechanism/DP_training/DPMLBench/algorithms/dp_sgd/main.py',
    'loss': 'dp_mechanism/DP_training/DPMLBench/algorithms/loss/main.py',
}

def main():
    parser = argparse.ArgumentParser(
        description='Unified training launcher for all main algorithms in this repository.'
    )
    parser.add_argument('--algo', type=str, required=True, choices=ALGO_PATHS.keys(), help='Algorithm to run')
    parser.add_argument('--net', type=str, default='resnet', help='Network architecture (e.g., resnet)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (e.g., cifar10)')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--batchsize', type=int, default=256, help='Batch size')
    parser.add_argument('--epoch', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--private', action='store_true', help='Enable differential privacy')
    parser.add_argument('--clip', type=float, default=4.0, help='Gradient clipping bound')
    parser.add_argument('--eps', type=float, default=None, help='Privacy parameter epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy parameter delta')
    parser.add_argument('--extra', type=str, default=None, help='Extra argument (e.g., clip_only)')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('other_args', nargs=argparse.REMAINDER, help='Other arguments to pass through')

    args = parser.parse_args()

    script_path = ALGO_PATHS[args.algo]
    cmd = [sys.executable, script_path,
           '--net', args.net,
           '--dataset', args.dataset,
           '--data_root', args.data_root,
           '--batchsize', str(args.batchsize),
           '--epoch', str(args.epoch),
           '--lr', str(args.lr),
           '--seed', str(args.seed),
           '--momentum', str(args.momentum),
           '--weight_decay', str(args.weight_decay),
           '--clip', str(args.clip),
           '--delta', str(args.delta)]
    if args.private:
        cmd.append('-p')
    if args.eps is not None:
        cmd.extend(['--eps', str(args.eps)])
    if args.extra is not None:
        cmd.extend(['--extra', args.extra])
    if args.other_args:
        cmd.extend(args.other_args)

    print('Running command:')
    print(' '.join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main() 