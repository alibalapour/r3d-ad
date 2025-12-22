"""
Training script for R3DAD with Pseudo Anomaly Synthesis.

This script integrates the pseudo anomaly synthesis module from preprocessing.py
with the R3DAD training pipeline. It trains the autoencoder model using synthetically
generated anomalies on normal point clouds.
"""

import os
import sys
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from easydict import EasyDict

from preprocessing import Dataset
from utils.misc import *
from models.autoencoder import *
from evaluation import ROC_AP


# Arguments
parser = argparse.ArgumentParser(description='Train R3DAD with Pseudo Anomaly Synthesis')

# Model arguments
parser.add_argument('--model', type=str, default='AutoEncoder')
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Dataset and preprocessing arguments
parser.add_argument('--category', type=str, default='ashtray0',
                    help='Category name from shapenet-ad dataset')
parser.add_argument('--dataset_path', type=str, default='./data/shapenet-ad')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Training batch size')
parser.add_argument('--rollout_batch_size', type=int, default=1,
                    help='Rollout batch size for RL-based training')
parser.add_argument('--num_works', type=int, default=4,
                    help='Number of dataloader workers')
parser.add_argument('--data_repeat', type=int, default=1,
                    help='Number of times to repeat training data')
parser.add_argument('--voxel_size', type=float, default=0.05,
                    help='Voxel size for sparse quantization')
parser.add_argument('--mask_num', type=int, default=32,
                    help='Number of spherical patches for anomaly synthesis')
parser.add_argument('--cache_dataset', type=eval, default=False, choices=[True, False],
                    help='Cache training dataset in memory')
parser.add_argument('--cache_test_set', type=eval, default=None,
                    help='Cache test dataset in memory (default: same as cache_dataset)')

# Pseudo anomaly synthesis parameters
parser.add_argument('--smart_anomaly', type=eval, default=True, choices=[True, False],
                    help='Use smart anomaly synthesis with configurable presets')
parser.add_argument('--R_low_bound', type=float, default=0.10,
                    help='Lower bound for anomaly radius (fraction of diameter)')
parser.add_argument('--R_up_bound', type=float, default=0.30,
                    help='Upper bound for anomaly radius')
parser.add_argument('--R_alpha', type=float, default=2.0,
                    help='Beta distribution alpha parameter for radius sampling')
parser.add_argument('--R_beta', type=float, default=5.0,
                    help='Beta distribution beta parameter for radius sampling')
parser.add_argument('--B_low_bound', type=float, default=0.02,
                    help='Lower bound for anomaly magnitude (displacement)')
parser.add_argument('--B_up_bound', type=float, default=0.15,
                    help='Upper bound for anomaly magnitude')
parser.add_argument('--B_alpha', type=float, default=2.0,
                    help='Beta distribution alpha parameter for magnitude sampling')
parser.add_argument('--B_beta', type=float, default=5.0,
                    help='Beta distribution beta parameter for magnitude sampling')
parser.add_argument('--one_sided_prob', type=float, default=0.7,
                    help='Probability of one-sided anomalies vs double-sided')
parser.add_argument('--cosine_kernel_prob', type=float, default=0.4,
                    help='Probability of using cosine kernel')
parser.add_argument('--gaussian_kernel_prob', type=float, default=0.3,
                    help='Probability of using gaussian kernel')
parser.add_argument('--poly_kernel_prob', type=float, default=0.2,
                    help='Probability of using polynomial kernel')
parser.add_argument('--hard_kernel_prob', type=float, default=0.1,
                    help='Probability of using hard kernel')
parser.add_argument('--poly_q', type=float, default=2.0,
                    help='Exponent for polynomial kernel')

# Validation
parser.add_argument('--validation', type=eval, default=False, choices=[True, False],
                    help='Enable validation during training')
parser.add_argument('--validation_suffixes', type=str, default='',
                    help='Comma-separated list of test file suffixes to use for validation')

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*1000)
parser.add_argument('--sched_end_epoch', type=int, default=300*1000)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--manual_seed', type=int, default=None,
                    help='Manual seed for reproducibility (overrides --seed)')
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_pseudo_anomaly')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_iters', type=int, default=40000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)

args = parser.parse_args()

# Set manual seed if provided, otherwise use seed
if args.manual_seed is not None:
    seed_all(args.manual_seed)
else:
    seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix=args.category + '_', 
                              postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

logger.info(args)
logger.info('Using Pseudo Anomaly Synthesis Module')

# Create EasyDict config for Dataset
cfg = EasyDict({
    'category': args.category,
    'batch_size': args.batch_size,
    'rollout_batch_size': args.rollout_batch_size,
    'num_works': args.num_works,
    'data_repeat': args.data_repeat,
    'voxel_size': args.voxel_size,
    'mask_num': args.mask_num,
    'cache_dataset': args.cache_dataset,
    'cache_test_set': args.cache_test_set,
    'validation': args.validation,
    'validation_suffixes': args.validation_suffixes,
    'smart_anomaly': args.smart_anomaly,
    'R_low_bound': args.R_low_bound,
    'R_up_bound': args.R_up_bound,
    'R_alpha': args.R_alpha,
    'R_beta': args.R_beta,
    'B_low_bound': args.B_low_bound,
    'B_up_bound': args.B_up_bound,
    'B_alpha': args.B_alpha,
    'B_beta': args.B_beta,
    'one_sided_prob': args.one_sided_prob,
    'cosine_kernel_prob': args.cosine_kernel_prob,
    'gaussian_kernel_prob': args.gaussian_kernel_prob,
    'poly_kernel_prob': args.poly_kernel_prob,
    'hard_kernel_prob': args.hard_kernel_prob,
    'poly_q': args.poly_q,
    'manual_seed': args.manual_seed if args.manual_seed is not None else args.seed,
})

logger.info('Loading datasets with pseudo anomaly synthesis...')
dataset = Dataset(cfg)
dataset.trainLoader()
dataset.testLoader()
if cfg.validation:
    dataset.valLoader()

logger.info(f'Training samples: {len(dataset.train_file_list)}')
logger.info(f'Test samples: {len(dataset.test_file_list)}')
if cfg.validation and dataset.validation_file_list:
    logger.info(f'Validation samples: {len(dataset.validation_file_list)}')
logger.info(f'Number of anomaly presets: {dataset.num_presets}')

# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = getattr(sys.modules[__name__], args.model)(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = getattr(sys.modules[__name__], args.model)(args).to(args.device)
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Training function
def train(it, train_loader):
    """Train one iteration."""
    # Get batch from dataloader
    try:
        batch = next(train_iter)
    except StopIteration:
        # Reset iterator if exhausted
        train_iter = iter(train_loader)
        batch = next(train_iter)
    
    # Extract data from batch
    # The preprocessing module returns voxelized data and original point clouds
    xyz_shifted = batch['xyz_shifted'].to(args.device)  # Anomalous point cloud
    xyz_original = batch['xyz_original'].to(args.device)  # Normal point cloud
    
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    
    # Forward pass: reconstruct the anomalous point cloud
    # The model should learn to reconstruct back to normal
    loss = model.get_loss(xyz_shifted)
    
    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | LR %.6f' % 
                (it, loss.item(), orig_grad_norm, optimizer.param_groups[0]['lr']))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()
    
    return loss.item()


def validate(it):
    """Validate the model on test set."""
    all_ref = []
    all_recons = []
    all_label = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset.test_data_loader, desc='Validate')):
            if args.num_val_batches > 0 and i >= args.num_val_batches:
                break
            
            ref = batch['xyz_original'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            # Encode and decode
            code = model.encode(ref)
            recons = model.decode(code, ref.size(0), flexibility=args.flexibility)
            
            all_ref.append(ref.cpu())
            all_recons.append(recons.cpu())
            all_label.append(labels.cpu())
    
    all_ref = torch.cat(all_ref, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    all_label = torch.cat(all_label, dim=0)
    
    # Create dummy mask (all zeros) since we don't have ground truth masks yet
    all_mask = torch.zeros_like(all_ref[:, :, 0])
    
    # Compute metrics
    try:
        metrics = ROC_AP(all_ref, all_recons, all_label, all_mask)
        roc_i = metrics.get('ROC_i', torch.tensor(0.0)).item()
        roc_p = metrics.get('ROC_p', torch.tensor(0.0)).item()
        ap_i = metrics.get('AP_i', torch.tensor(0.0)).item()
        ap_p = metrics.get('AP_p', torch.tensor(0.0)).item()
        
        logger.info('[Val] Iter %04d | ROC_i %.6f | ROC_p %.6f | AP_i %.6f | AP_p %.6f' % 
                    (it, roc_i, roc_p, ap_i, ap_p))
        
        writer.add_scalar('val/ROC_i', roc_i, it)
        writer.add_scalar('val/ROC_p', roc_p, it)
        writer.add_scalar('val/AP_i', ap_i, it)
        writer.add_scalar('val/AP_p', ap_p, it)
        writer.flush()
        
        return roc_i
    except Exception as e:
        logger.warning(f'[Val] Error computing metrics: {e}')
        return 0.0


# Main training loop
logger.info('Starting training with pseudo anomaly synthesis...')
train_iter = iter(dataset.train_data_loader)

try:
    it = 1
    while it <= args.max_iters:
        train_loss = train(it, dataset.train_data_loader)
        
        if it % args.val_freq == 0 or it == args.max_iters:
            score = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, score, opt_states, it)
        
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
    
logger.info('Training completed!')
