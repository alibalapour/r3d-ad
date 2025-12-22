"""
Wrapper script to train R3DAD with pseudo anomaly synthesis across categories.

This script runs training with pseudo anomaly synthesis for one or all categories
in the shapenet-ad dataset.
"""

import argparse
import os
import time
from pathlib import Path

from utils.config import cmd_from_config
from utils.dataset import all_shapenetad_cates


def main(args):
    exp_name = Path(args.config).stem
    time_fix = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    cfg_cmd = cmd_from_config(args.config)
    
    # Determine categories to train on
    if args.category == 'all':
        cates = all_shapenetad_cates
    else:
        cates = [args.category]
    
    for cate in cates:
        log_dir = f'logs_pseudo_anomaly/{exp_name}_{time_fix}' + (f'_{args.tag}' if args.tag else '') + '/'
        cmd = f"python train_with_pseudo_anomaly.py --category {cate} --log_root {log_dir}" + cfg_cmd
        
        print(f"\n{'='*80}")
        print(f"Training category: {cate}")
        print(f"Command: {cmd}")
        print(f"{'='*80}\n")
        
        os.system(cmd)
        
        if args.single:
            print(f"\nCompleted training for category: {cate}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train R3DAD with pseudo anomaly synthesis')
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument('--category', type=str, default='all',
                        help='Category to train (default: all)')
    parser.add_argument('--tag', type=str, default='',
                        help='Tag to append to log directory name')
    parser.add_argument('--single', action='store_true',
                        help='Train only the first category (for testing)')
    args = parser.parse_args()
    main(args)
