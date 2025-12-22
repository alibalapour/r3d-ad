import multiprocessing as mp
import math
import glob
import torch
import random
import numpy as np
import open3d as o3d
import scipy.ndimage
import scipy.interpolate
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import data.AnomalyShapeNet.transform as aug_transform
import os
import re
from pathlib import Path

from dataclasses import dataclass, asdict
from typing import Callable


from utils.visualize import save_pc_plotly_html
DEBUG = False


def _rand_sign(p_plus=0.5):
    """+1 with prob p_plus, else -1."""
    return +1 if np.random.rand() < float(p_plus) else -1


class AnomalyPreset:
    """
    Factory for anomaly configs. Reads parameter ranges from `args`.

    Expected args fields (with sensible fallbacks used via getattr):
      - R_low_bound, R_up_bound, R_alpha, R_beta
      - B_low_bound, B_up_bound, B_alpha, B_beta
      - p_bulge (for ripple), micro_count (how many dimples to spawn if you loop externally)
    """

    def __init__(self, args):
        self.args = args
        self.presets = [
            self.type_1_basic_bulge,
            self.type_2_basic_dent,
            self.type_3_ridge,
            self.type_4_trench,
            self.type_5_elliptic_patch_flat_spot,
            self.type_6_skewed_impact_crater,
            # (your #3 in text: shear/slip u)
            self.type_7_shear_u,
            # (your #3 in text: shear/slip v)
            self.type_7b_shear_v,
            self.type_8_double_sided_ripple,
            # base config (apply many times externally)
            self.type_9_micro_dimple_field_base,
            self.type_10_directional_drag_stretch
        ]

    # ----------------------
    # Helpers
    # ----------------------
    def get_R_B(self):
        R = self.args.R_low_bound + (self.args.R_up_bound - self.args.R_low_bound) * \
            np.random.beta(self.args.R_alpha, self.args.R_beta, size=1)
        B = self.args.B_low_bound + (self.args.B_up_bound - self.args.B_low_bound) * \
            np.random.beta(self.args.B_alpha, self.args.B_beta, size=1)
        return float(R), float(B)

    def _p(self, name, default):
        return getattr(self.args, name, default)

    # ----------------------
    # 1) Basic Local Bulge (isotropic, outward)
    # ----------------------
    def type_1_basic_bulge(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 1.0),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=+1,         # bulge
            beta=B,
            sigma=0.35
        )

    # ----------------------
    # 2) Basic Local Dent (isotropic, inward)
    # ----------------------
    def type_2_basic_dent(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 1.0),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,         # dent
            beta=B,
            sigma=0.35
        )

    # ----------------------
    # (Spec #2) Elongated Ridge (anisotropic bulge along u)
    # ----------------------
    def type_3_ridge(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(2.5, 0.7, 0.6),     # elongated along u
            kernel="gaussian",
            dir_mode="normal_mean",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0, 0, 1]),
            alpha=+1,                  # ridge
            beta=B,
            sigma=0.4
        )

    # ----------------------
    # (Spec #2) Elongated Groove (anisotropic dent along u)
    # ----------------------
    def type_4_trench(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(3.0, 0.7, 0.5),
            kernel="cosine",
            dir_mode="normal_mean",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0, 0, 1]),
            alpha=-1,                  # trench
            beta=B
        )

    # ----------------------
    # 5) Elliptic Patch / Flat Spot (pressed region)  — variant A (flatten)
    # ----------------------
    def type_5_elliptic_patch_flat_spot(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.3, 1.0, 0.5),
            kernel="gaussian",
            dir_mode="normal_point",
            one_sided=True,
            gate_mode="normals",
            alpha=-1,      # press/flatten
            beta=B,
            sigma=0.5
        )

    # ----------------------
    # 6) Skewed Impact Crater (oblique one-sided dent)
    # ----------------------
    def type_6_skewed_impact_crater(self):
        R, B = self.get_R_B()
        gate_offset = self._p("gate_offset", 0.05)
        gate_sharpness = self._p("gate_sharpness", 8.0)
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 0.6),
            kernel="cosine",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0.2, 0.3, 0.93]),  # oblique
            gate_offset=gate_offset,
            gate_sharpness=gate_sharpness,
            dir_mode="normal_point",
            alpha=-1,      # impact dent
            beta=B
        )

    # ----------------------
    # 3) Shear / Slip along u (tangential)
    # ----------------------
    def type_7_shear_u(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.8),
            kernel="cosine",
            dir_mode="tangent_u",   # slip along u
            one_sided=False,
            alpha=+1,
            beta=B
        )

    # ----------------------
    # 3) Shear / Slip along v (tangential, opposite)
    # ----------------------
    def type_7b_shear_v(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.8),
            kernel="cosine",
            dir_mode="tangent_v",   # slip along v
            one_sided=False,
            alpha=-1,
            beta=B
        )

    # ----------------------
    # 4) Double-Sided Ripple (cosine, alternating sign)
    # ----------------------
    def type_8_double_sided_ripple(self):
        R, B = self.get_R_B()
        p_bulge = self._p("p_bulge", 0.5)
        alpha = _rand_sign(p_bulge)   # random ±1 per call
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.0, 1.0, 0.8),
            kernel="cosine",
            one_sided=False,           # double-sided ripple
            dir_mode="normal_mean",
            alpha=alpha,
            beta=B
        )

    # ----------------------
    # 9) Micro Dimple Field (single tiny dimple config; apply N times externally)
    # ----------------------
    def type_9_micro_dimple_field_base(self):
        # For corrosion/pitting, you’ll likely loop this preset at random centers.
        # Keep B very small so each dimple is subtle.
        _, _ = self.get_R_B()  # ignore R; micro uses local radii/sigma primarily
        B = self._p("micro_beta", 0.005)
        return SmartAnomaly_Cfg(
            radii=(0.4, 0.4, 0.4),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,      # tiny dent
            beta=B,
            sigma=0.4
        )

    # ----------------------
    # 10) Directional Drag / Stretch (anisotropic, gated)
    # ----------------------
    def type_10_directional_drag_stretch(self):
        R, B = self.get_R_B()
        # Slightly smaller beta to look more like plastic deformation
        beta_scaled = B * self._p("drag_beta_scale", 0.17)  # ~0.025 if B~0.15
        return SmartAnomaly_Cfg(
            R=R,
            radii=(2.0, 0.8, 0.6),
            kernel="gaussian",
            dir_mode="tangent_u",
            one_sided=True,
            gate_mode="global",
            n_global=np.array([0, 1, 0]),
            alpha=+1,
            beta=beta_scaled,
            sigma=0.5
        )


# To inject params into dataloader workers
manager = mp.Manager()
standard_param_queue = manager.Queue()
rollout_param_queue = manager.Queue()
# Backwards compatibility: retain the original name for the standard queue
param_queue = standard_param_queue


# Contribution: Writing dataloader collate function as a closure to capture shared config that can be modified externally through training loop, without any signi


@dataclass
class SmartAnomaly_Cfg:
    # size & strength
    R: float = None            # support radius; None -> 0.2 * object diameter
    beta: float = 0.08                # magnitude (your distance_to_move)
    alpha: int = None          # +1 bulge, -1 cavity, None -> random w.p. p_bulge
    p_bulge: float = 0.5

    # falloff kernel
    kernel: str = "cosine"            # {"cosine","gaussian","poly","hard"}
    q: float = 2.0
    sigma: float = 0.35               # as fraction of R for gaussian

    # anisotropy (ellipsoid radii along local frame axes)
    radii: tuple = (1.0, 1.0, 1.0)    # (ru, rv, rn); 1,1,1 == sphere

    # displacement direction
    # {"normal_point","normal_mean","tangent_u","tangent_v"}
    dir_mode: str = "normal_point"

    # gating (NEW): keep deformation on one side only
    one_sided: bool = True                    # turn on to gate to one side of a plane
    gate_mode: str = "normals"                 # {"global","normals"}
    # global outward direction (used when gate_mode="global")
    n_global: tuple = (0.0, 0.0, 1.0)
    gate_soft: bool = True                     # soft logistic gate vs hard step
    gate_sharpness: float = 30.0               # larger = crisper
    # shift plane along n (push gate outward)
    gate_offset: float = 0.0

    # extras
    # 0..1 probability of removing center points (holes)
    carve_strength: float = 0.0
    smooth_steps: int = 0             # Laplacian smoothing steps inside support
    smooth_lambda: float = 0.15
    seed: int = None


@dataclass
class CollateBundle:
    """Container for the different training collate entrypoints."""

    standard: Callable
    rollout: Callable
    dispatch: Callable

    def __call__(self, batch_indices):
        return self.dispatch(batch_indices)


def make_collate(dataset_object, param_queue):
    import queue as _queue
    import numpy as np
    import torch
    import MinkowskiEngine as ME
    import open3d as o3d

    default_beta = 0.08
    queue_timeout = 2

    def _get_params():
        """ Reads a dict from the queue and ensures there’s at least a 'beta'

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        try:
            params = param_queue.get(timeout=queue_timeout)
        except _queue.Empty:
            params = {}
        if not isinstance(params, dict):
            raise TypeError(f"Expected parameter dict, got {type(params)!r}")
        params.setdefault('beta', default_beta)
        return params

    # Seed for deterministic preset sampling; falls back to torch worker seed when
    # a manual seed is not provided in the config.
    fallback_seed = getattr(dataset_object.global_cfg, 'manual_seed', None)
    # Cache of per-worker RNGs so each DataLoader worker reuses its own generator.
    worker_rngs = {}

    def _get_fallback_rng():
        """Return a per-worker RNG seeded deterministically."""

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else None
        rng = worker_rngs.get(worker_id)
        if rng is None:
            # Use manual seed when set so experiments are reproducible; otherwise
            # mirror PyTorch's worker seed so the RNG stays stable per worker.
            if fallback_seed is not None:
                seed = int(fallback_seed)
            else:
                seed = int(torch.initial_seed())
            rng = np.random.default_rng(seed)
            # Persist the generator for this worker id to avoid reseeding mid-epoch.
            worker_rngs[worker_id] = rng
        return rng

    num_presets = max(int(getattr(dataset_object, 'num_presets', 0)), 1)

    def _sample_random_actions(num_actions=1):
        """Sample random preset indices when none are provided."""

        if num_presets <= 0:
            raise ValueError("Dataset has no anomaly presets configured.")
        num_actions = max(int(num_actions), 1)
        rng = _get_fallback_rng()
        return rng.integers(0, num_presets, size=(num_actions,), dtype=np.int64)

    def _normalize_rollout_actions(params):
        """Ensure rollout actions is a 1-D array of preset indices."""

        actions = params.get('actions') if params is not None else None
        if actions is None:
            num_actions = 1
            if isinstance(params, dict):
                for key in ('num_actions', 'rollout_K', 'K'):
                    maybe = params.get(key)
                    if maybe is not None:
                        try:
                            num_actions = max(int(maybe), 1)
                        except (TypeError, ValueError):
                            num_actions = 1
                        break
            actions = _sample_random_actions(num_actions)
        else:
            actions = np.asarray(actions)
            if actions.ndim == 0:
                actions = actions.reshape(1)
            elif actions.ndim == 2 and actions.shape[1] == 1:
                actions = actions.reshape(-1)
        if actions.ndim != 1:
            raise ValueError(
                f"`actions` must be a 1-D array of preset indices, got shape {actions.shape}")
        return actions.astype(np.int64, copy=False)

    def _select_best_action(params):
        """Select the preset index to apply for standard batches."""

        actions = params.get('actions')
        if actions is None:
            return _sample_random_actions(num_actions=1)

        actions = np.asarray(actions)
        if actions.ndim == 0:
            actions = actions.reshape(1)
        elif actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions.reshape(-1)
        elif actions.ndim != 1:
            raise ValueError(
                f"`actions` must be 1-D when provided, got shape {actions.shape}")

        actions = actions.astype(np.int64, copy=False)
        rewards = params.get('rewards', None)
        if rewards is not None:
            rewards = np.asarray(rewards).reshape(-1)
            if rewards.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"`rewards` length ({rewards.shape}) does not match actions ({actions.shape})")
            best_idx = int(np.argmax(rewards))
        else:
            best_idx = int(params.get('best_action_idx', 0))

        if not 0 <= best_idx < actions.shape[0]:
            raise ValueError(
                f"`best_action_idx` {best_idx} out of range for {actions.shape[0]} actions")

        return actions[best_idx:best_idx + 1]

    def _collate_impl(id_list, params, actions_arr, rollout_mode, N_fixed: int = 2048, generator: torch.Generator = None):
        actions_arr = np.asarray(actions_arr, dtype=np.int64)
        if actions_arr.ndim != 1:
            raise ValueError(
                f"Expected actions of shape (N,), got {actions_arr.shape}")

        # --- RNG for reproducibility (optional) ---
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(getattr(dataset_object.global_cfg, 'manual_seed', None)))  # uncomment for deterministic sampling

        beta_override = params.get('beta', default_beta)
        beta_override = None if beta_override is None else float(beta_override)

        sample_indices = list(id_list)
        if rollout_mode:
            force_single_sample = bool(params.get('force_single_sample', True))
            if force_single_sample and len(sample_indices) > 1:
                sample_indices = [sample_indices[0]]
            rollout_voxel_scale = float(params.get('rollout_voxel_scale', 1.0))
            this_voxel_size = float(
                dataset_object.voxel_size) * rollout_voxel_scale
        else:
            this_voxel_size = float(dataset_object.voxel_size)

        max_points_per_group = params.get('max_points_per_group', None)
        if max_points_per_group is not None:
            max_points_per_group = int(max_points_per_group)

        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        xyz_shifted = []
        v2p_index_batch = []
        total_voxel_num = 0
        batch_count = [0]
        total_point_num = 0
        gt_offset_list = []

        # --- keep per-item tensors for BN3 outputs for point cloud feature extraction ---
        xyz_original_per_item = []
        xyz_shifted_per_item = []
        offset_per_item = []
        
        # Track shift_indices for both rollout and standard modes
        shift_indices_list = []
        
        # Store patch point clouds for FPFH feature extraction
        patch_xyz_list = []

        if rollout_mode:
            split_sizes = []
            action_map = []
            anomaly_cfg_records = []
            preset_indices = []

        for idx in sample_indices:    # This loop is on point clouds, so in each iteration we are working with one point cloud (In rollout mode, there will be only one point cloud with multiple actions)
            # Set name variables
            fn_path = dataset_object.train_file_list[idx]
            base_name = dataset_object.train_file_list[idx]

            # Load point cloud and preprocess
            coord, vertex_normals = dataset_object._load_train_point_cloud(fn_path)
            mask = np.ones(coord.shape[0]) * -1                  # Initialize mask as -1 (no mask)

            # Apply any dataset-specific augmentations and compositions
            Point_dict = {
                'coord': coord,
                'normal': vertex_normals,
                'mask': mask
            }
            Point_dict, centers = dataset_object.train_aug_compose(Point_dict)

            # Extract base data arrays
            xyz_base = Point_dict['coord'].astype(np.float32)
            normal_base = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)
            
            # Transform mask values to expected range by giving valid values to mask indicies which are out of bound
            mask[mask == (dataset_object.mask_num + 1)
                 ] = dataset_object.mask_num - 1                 

            action_count = actions_arr.shape[0] if rollout_mode else 1
            
            # For loop over actions (in rollout mode) or single action (standard mode)
            for action_idx in range(action_count):
                
                # Get index of preset to use
                preset_idx = int(actions_arr[action_idx]) if rollout_mode else int(
                    actions_arr[0])
                if not 0 <= preset_idx < len(dataset_object.anomaly_presets):
                    raise ValueError(
                        f"Preset index {preset_idx} out of range (have {len(dataset_object.anomaly_presets)})"
                    )

                # Instantiate anomaly config from preset
                preset_cfg = dataset_object.anomaly_presets[preset_idx]()
                cfg_copy = SmartAnomaly_Cfg(**asdict(preset_cfg))
                if beta_override is not None:
                    cfg_copy.beta = beta_override
                anomlay_cfg = cfg_copy

                # Randomly select indices for shifting and extract corresponding data from xyz_base, normal_base, and centers.
                num_shift = 1
                mask_range = np.arange(0, dataset_object.mask_num // 2)   # TODO: why half of mask_num? (the code is based on PO3AD codebase)
                shift_index = np.random.choice(
                    mask_range, num_shift, replace=False)
                
                
                 # Use pre-fetched shift_index if available and enabled
                if (dataset_object.use_prefetched_shift_indices and 
                    dataset_object.prefetched_shift_indices is not None and 
                    idx in dataset_object.prefetched_shift_indices):
                    shift_index = dataset_object.prefetched_shift_indices[idx]
                else:
                    shift_index = np.random.choice(
                        mask_range, num_shift, replace=False)
                
                
                
                mask_used = mask.copy()
                mask_used[np.isin(mask_used, shift_index)] = -1
                shift_xyz_base = xyz_base[mask_used == -1].copy()
                shift_normal_base = normal_base[mask_used == -1].copy()
                center = centers[shift_index[0]]

                # Prepare data for anomaly generation
                xyz = xyz_base.copy()
                normal = normal_base.copy()
                mlocal = mask_used
                shift_xyz = shift_xyz_base
                shift_normal = shift_normal_base

                cfg_dict = asdict(anomlay_cfg)
                radii_val = cfg_dict.get('radii')
                if isinstance(radii_val, (list, tuple)):
                    cfg_dict['radii'] = [float(x) for x in radii_val]
                elif radii_val is None:
                    cfg_dict['radii'] = []
                else:
                    cfg_dict['radii'] = [float(radii_val)]

                n_global_val = [0, 0, 1]
                cfg_dict['action_index'] = int(
                    action_idx) if rollout_mode else 0
                cfg_dict['preset_index'] = int(preset_idx)
                if beta_override is not None:
                    cfg_dict['beta_override'] = float(beta_override)
                R_value = cfg_dict.get('R')
                cfg_dict['R_value'] = float(
                    R_value) if R_value is not None else None

                # Generate pseudo-anomaly point cloud
                if dataset_object.global_cfg.smart_anomaly:
                    shifted_xyz = dataset_object.generate_pseudo_anomaly(
                        shift_xyz, shift_normal, center, anomlay_cfg=anomlay_cfg
                    )
                else:
                    shifted_xyz = dataset_object.generate_pseudo_anomaly_original(
                        shift_xyz, shift_normal, center
                    )

                new_xyz = xyz.copy()
                new_xyz[mlocal == -1] = shifted_xyz

                # Optionally limit number of points per group for rollout memory control
                if max_points_per_group is not None and new_xyz.shape[0] > max_points_per_group:
                    sel = np.random.choice(
                        new_xyz.shape[0], max_points_per_group, replace=False)
                    new_xyz = new_xyz[sel]
                    xyz = xyz[sel]

                # Compute ground-truth offset
                gt_offset = new_xyz - xyz  # (Ni, 3)

                # Refine file name
                if rollout_mode:
                    file_name.append(f"{base_name}::act{action_idx}")
                else:
                    file_name.append(base_name)

                # tensors per-item (for both legacy flat concat and BN3 sampling).
                xyz_t = torch.from_numpy(xyz)           # (Ni, 3)
                new_xyz_t = torch.from_numpy(new_xyz)   # (Ni, 3)
                gt_off_t = torch.from_numpy(gt_offset)  # (Ni, 3)

                xyz_original.append(xyz_t)
                xyz_shifted.append(new_xyz_t)
                gt_offset_list.append(gt_off_t)

                # Keep per-item copies for BN3 outputs. These will be used to build B*N_fixed*3 later.
                xyz_original_per_item.append(xyz_t)
                xyz_shifted_per_item.append(new_xyz_t)
                offset_per_item.append(gt_off_t)

                # Voxelization using MinkowskiEngine
                quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(
                    new_xyz, new_xyz,
                    quantization_size=this_voxel_size,
                    return_index=True,
                    return_inverse=True
                )

                v2p_index = inverse_index + total_voxel_num
                total_voxel_num += index.shape[0]

                total_point_num += inverse_index.shape[0]
                batch_count.append(total_point_num)

                xyz_voxel.append(quantized_coords)
                feat_voxel.append(feats_all)
                v2p_index_batch.append(v2p_index)
                
                # Track shift_index for both modes
                shift_indices_list.append(int(shift_index[0]))
                
                # Store patch point cloud for FPFH feature extraction in RL state
                patch_xyz_list.append(torch.from_numpy(shift_xyz_base.copy()).to(torch.float32))

                if rollout_mode:
                    split_sizes.append(int(inverse_index.shape[0]))
                    action_map.append(int(action_idx))
                    anomaly_cfg_records.append(cfg_dict)
                    preset_indices.append(int(preset_idx))


        # MinkowskiEngine collate
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)
        xyz_original_cat = torch.cat(xyz_original, 0).to(torch.float32)
        xyz_shifted_cat = torch.cat(xyz_shifted, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        batch_count = torch.from_numpy(np.array(batch_count, dtype=np.int64))
        batch_offset = torch.cat(gt_offset_list, 0).to(torch.float32)

        # ==================================================
        # Build fixed-size (B, N_fixed, 3) tensors for point cloud features extraction
        # ==================================================
        def _sample_fixed(idx_len: int, N: int, gen: torch.Generator, device: torch.device):
            if idx_len >= N:
                return torch.randperm(idx_len, generator=gen, device=device)[:N]
            # repeat-to-fill strategy
            num_full = N // idx_len
            rem = N - num_full * idx_len
            base = torch.arange(idx_len, device=device)
            if rem > 0:
                extra = torch.randint(low=0, high=idx_len, size=(
                    rem,), generator=gen, device=device)
                return torch.cat([base.repeat(num_full), extra], dim=0)
            else:
                return base.repeat(num_full)

        B = len(xyz_original_per_item)
        device = xyz_original_cat.device  # tensors above are on CPU; preserve that

        bn3_xyz_list = []
        bn3_xyz_shifted_list = []
        bn3_offset_list = []
        bn3_indices = []
        orig_lengths = []

        for i in range(B):
            pts = xyz_original_per_item[i].to(device)    # (Ni,3)
            pts_new = xyz_shifted_per_item[i].to(device)  # (Ni,3)
            pts_off = offset_per_item[i].to(device)      # (Ni,3)
            Ni = pts.shape[0]
            orig_lengths.append(Ni)

            idx_sel = _sample_fixed(Ni, N_fixed, generator, device)
            bn3_indices.append(idx_sel)

            bn3_xyz_list.append(pts[idx_sel].unsqueeze(0))       # (1,N,3)
            bn3_xyz_shifted_list.append(pts_new[idx_sel].unsqueeze(0))
            bn3_offset_list.append(pts_off[idx_sel].unsqueeze(0))

        xyz_bn3 = torch.cat(bn3_xyz_list, dim=0).to(
            torch.float32)             # (B,N,3)
        xyz_shifted_bn3 = torch.cat(bn3_xyz_shifted_list, dim=0).to(
            torch.float32)  # (B,N,3)
        offset_bn3 = torch.cat(bn3_offset_list, dim=0).to(
            torch.float32)       # (B,N,3)
        bn3_indices = torch.stack(bn3_indices, dim=0).to(
            torch.int64)          # (B,N)
        orig_lengths = torch.tensor(
            orig_lengths, dtype=torch.int32)           # (B,)

        out = {
            'xyz_voxel': xyz_voxel_batch,
            'feat_voxel': feat_voxel_batch,
            'xyz_original': xyz_original_cat,    # legacy flat concat
            'fn': file_name,
            'v2p_index': v2p_index_batch,
            'xyz_shifted': xyz_shifted_cat,
            'batch_count': batch_count,
            'batch_offset': batch_offset,

            # --- new BN3 tensors ---
            'xyz_bn3': xyz_bn3,                         # (B, N_fixed, 3)
            'xyz_shifted_bn3': xyz_shifted_bn3,         # (B, N_fixed, 3)
            'offset_bn3': offset_bn3,                   # (B, N_fixed, 3)
            # (B, N_fixed) per-item indices
            'bn3_indices': bn3_indices,
            'orig_lengths': orig_lengths,               # (B,)
            'N_fixed': int(N_fixed),
            # --- shift_indices for RL state ---
            # --- shift_indices for RL state ---
            'shift_indices': torch.as_tensor(shift_indices_list, dtype=torch.long),
            
            # --- patch point clouds for FPFH feature extraction ---
            'patch_xyz': patch_xyz_list,  # List of tensors, each of shape (N_patch_i, 3)
        }
        
        

        if rollout_mode:
            out.update({
                'split_sizes': torch.as_tensor(split_sizes, dtype=torch.long),
                'action_map': torch.as_tensor(action_map, dtype=torch.long),
                'anomaly_cfg_params': anomaly_cfg_records,
                'preset_indices': torch.as_tensor(preset_indices, dtype=torch.long),
            })

        return out

    def _resolve_mode(params, fallback="standard"):
        mode = params.get('collate_mode')
        if mode is not None:
            mode = str(mode).lower()
            if mode in {"rollout", "standard"}:
                return mode
            raise ValueError(
                f"Unknown collate_mode '{mode}'. Expected 'rollout' or 'standard'.")

        use_rollout_flag = params.get('use_rollout')
        if use_rollout_flag is not None:
            return 'rollout' if bool(use_rollout_flag) else 'standard'

        actions_hint = params.get('actions', None)
        if actions_hint is not None:
            actions_hint = np.asarray(actions_hint)
            if actions_hint.ndim == 2 and actions_hint.shape[0] > 1:
                return 'rollout'

        return fallback

    def _dispatch_with_mode(id_list, params, forced_mode=None):
        params_local = dict(params) if isinstance(
            params, dict) else dict(params)
        if forced_mode is not None:
            params_local['collate_mode'] = forced_mode

        mode = _resolve_mode(params_local)
        if mode == 'rollout':
            actions_arr = _normalize_rollout_actions(params_local)
            return _collate_impl(id_list, params_local, actions_arr, rollout_mode=True)
        if mode == 'standard':
            actions_arr = _select_best_action(params_local)
            return _collate_impl(id_list, params_local, actions_arr, rollout_mode=False)
        raise ValueError(
            f"Unknown collate_mode '{mode}'. Expected 'rollout' or 'standard'.")

    def _dispatch_auto(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params)

    def _dispatch_rollout(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params, forced_mode='rollout')

    def _dispatch_standard(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params, forced_mode='standard')

    return CollateBundle(
        standard=_dispatch_standard,
        rollout=_dispatch_rollout,
        dispatch=_dispatch_auto,
    )


class Dataset:
    def __init__(self, cfg):
        self.global_cfg = cfg
        self.batch_size = cfg.batch_size
        self.rollout_batch_size = getattr(cfg, 'rollout_batch_size', 1)
        self.dataset_workers = cfg.num_works
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num
        cache_dataset = getattr(cfg, 'cache_dataset', False)
        cache_test_set = getattr(cfg, 'cache_test_set', None)
        self.cache_dataset = bool(cache_dataset)
        self.cache_test_set = self.cache_dataset if cache_test_set is None else bool(cache_test_set)
        self._train_cache = {}
        self._test_cache = {}
        self.standard_param_queue = standard_param_queue
        self.rollout_param_queue = rollout_param_queue
        self.anomaly_presets = AnomalyPreset(self.global_cfg).presets
        self.num_presets = len(self.anomaly_presets)

        self.category = cfg.category
        self.category_list = self._list_categories()
        assert self.category in self.category_list

        self.train_file_list = self._build_train_file_list()
        self.test_file_list = self._build_test_file_list()
        self.validation_suffixes = self._parse_validation_suffixes(
            getattr(cfg, 'validation_suffixes', ''))
        if self.global_cfg.validation:
            self.validation_file_list = self._build_validation_file_list()
        else:
            self.validation_file_list = []
        self._eval_file_list = self.test_file_list

        self.normal_tag = getattr(cfg, 'normal_tag', 'positive')
        self.gt_delimiter = getattr(cfg, 'gt_delimiter', ',')
        self.gt_mask_dir = self._default_gt_mask_dir()

        transform_module = self._get_transform_module()
        self.NormalizeCoord = transform_module.NormalizeCoord()
        self.CenterShift = transform_module.CenterShift(apply_z=True)
        self.RandomRotate_z = transform_module.RandomRotate(
            angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = transform_module.RandomRotate(
            angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = transform_module.RandomRotate(
            angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = transform_module.SphereCropMask(
            part_num=self.mask_num)

        self.train_aug_compose = transform_module.Compose([self.CenterShift, self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
                                                        self.NormalizeCoord, self.SphereCropMask])

        self.test_aug_compose = transform_module.Compose(
            [self.CenterShift, self.NormalizeCoord])
        
                
        # Initialize storage for pre-fetched shift indices
        self.prefetched_shift_indices = None
        self.use_prefetched_shift_indices = getattr(cfg, 'use_prefetched_shift_indices', True)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

    # ------------------------------------------------------------------
    # Dataset configuration helpers (override in subclasses)
    # ------------------------------------------------------------------
    def _get_transform_module(self):
        """Return the transform module used to build augmentation pipelines."""
        return aug_transform

    def _list_categories(self):
        root = Path('data/AnomalyShapeNet/dataset/pcd')
        if not root.exists():
            return []
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    def _train_file_glob(self):
        return str(Path('data/AnomalyShapeNet/dataset/obj') / self.category / '*.obj')

    def _train_file_filter(self, candidates):
        pattern = re.compile(r'template')
        return sorted([fn for fn in candidates if pattern.search(fn)])

    def _build_train_file_list(self):
        data_list = glob.glob(self._train_file_glob())
        train_files = self._train_file_filter(data_list)
        return train_files * self.data_repeat

    def _test_file_glob(self):
        return str(Path('data/AnomalyShapeNet/dataset/pcd') / self.category / 'test' / '*.pcd')

    def _build_test_file_list(self):
        test_files = glob.glob(self._test_file_glob())
        test_files.sort()
        return test_files

    def _parse_validation_suffixes(self, suffixes_raw):
        if suffixes_raw is None:
            return None
        if isinstance(suffixes_raw, (list, tuple, set)):
            parsed = {int(s) for s in suffixes_raw}
            return parsed if parsed else None
        suffixes_str = str(suffixes_raw).strip()
        if suffixes_str.lower() in {'', 'auto', 'none'}:
            return None
        try:
            parts = suffixes_str.split(',')
            parsed = {int(p.strip()) for p in parts if p.strip()}
            return parsed if parsed else None
        except ValueError:
            return None

    def _extract_suffix_index(self, path_str: str):
        match = re.search(r'(\d+)$', Path(path_str).stem)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _build_validation_file_list(self):
        if not self.test_file_list:
            return []

        indexed = []
        for fn in self.test_file_list:
            idx = self._extract_suffix_index(fn)
            if idx is not None:
                indexed.append((idx, fn))

        if not indexed:
            return []

        available_suffixes = sorted({idx for idx, _ in indexed})
        target_suffixes = set()
        if self.validation_suffixes:
            target_suffixes = {idx for idx in available_suffixes if idx in self.validation_suffixes}
        if not target_suffixes:
            target_suffixes = set(available_suffixes[:2])

        val_files = [fn for idx, fn in indexed if idx in target_suffixes]
        val_files.sort()

        print(f"Validation samples ({len(val_files)}) using suffixes {sorted(target_suffixes)}:")
        for fn in val_files:
            print(f"  - {Path(fn).name}")

        return val_files

    def _default_gt_mask_dir(self):
        return Path('data/AnomalyShapeNet/dataset/pcd') / self.category / 'GT'

    def _resolve_gt_path(self, sample_name: str) -> Path:
        return Path(self.gt_mask_dir) / f'{sample_name}.txt'

    def _read_anomalous_points(self, gt_path: Path) -> np.ndarray:
        kwargs = {}
        if self.gt_delimiter is not None:
            kwargs['delimiter'] = self.gt_delimiter
        data = np.loadtxt(gt_path, **kwargs)
        return data[:, 0:3]

    def _load_normal_point_cloud(self, fn_path: str) -> np.ndarray:
        if self.cache_test_set:
            cached = self._test_cache.get((fn_path, 'normal'))
            if cached is not None:
                return cached.copy()
        pcd = o3d.io.read_point_cloud(fn_path)
        points = np.asarray(pcd.points)
        if self.cache_test_set:
            self._test_cache[(fn_path, 'normal')] = points.copy()
        return points

    def _load_anomalous_point_cloud(self, fn_path: str) -> np.ndarray:
        sample_name = Path(fn_path).stem
        gt_path = self._resolve_gt_path(sample_name)
        cache_key = (fn_path, 'anomaly')
        if self.cache_test_set:
            cached = self._test_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        points = self._read_anomalous_points(gt_path)
        if self.cache_test_set:
            self._test_cache[cache_key] = points.copy()
        return points

    def _load_train_point_cloud(self, fn_path: str):
        if self.cache_dataset:
            cached = self._train_cache.get(fn_path)
            if cached is not None:
                coord_cached, normal_cached = cached
                return coord_cached.copy(), normal_cached.copy()

        obj = o3d.io.read_triangle_mesh(fn_path)
        obj.compute_vertex_normals()                         # Compute normals
        coord = np.asarray(obj.vertices)                     # Extract vertices (N, 3)
        vertex_normals = np.asarray(obj.vertex_normals)      # Extract normals (N, 3)

        if self.cache_dataset:
            self._train_cache[fn_path] = (coord.copy(), vertex_normals.copy())

        return coord, vertex_normals

    def _load_test_point_cloud(self, fn_path: str):
        is_normal = self.normal_tag and self.normal_tag in Path(fn_path).name
        if is_normal:
            coord = self._load_normal_point_cloud(fn_path)
            label = 0
        else:
            coord = self._load_anomalous_point_cloud(fn_path)
            label = 1
        return coord, label


    def prefetch_shift_indices(self, seed=None):
        """
        Pre-fetch shift indices for all training samples.
        
        This method generates shift_index values for each training sample ahead of time,
        which can be used during training for reproducibility and for RL state construction.
        
        Args:
            seed: Random seed for reproducibility. If None, uses the manual_seed from config,
                  or defaults to 42 if manual_seed is not set.
        
        Returns:
            dict: A dictionary mapping sample indices to their pre-fetched shift indices.
                  Format: {sample_idx: shift_index}
        """
        # Default seed for reproducibility when manual_seed is not configured
        DEFAULT_PREFETCH_SEED = 42
        
        if seed is None:
            seed = getattr(self.global_cfg, 'manual_seed', DEFAULT_PREFETCH_SEED)
        
        rng = np.random.default_rng(seed)
        
        # Calculate mask_range (same as in collate function)
        mask_range = np.arange(0, self.mask_num // 2)
        num_shift = 1
        
        # Pre-allocate dictionary for better performance
        num_samples = len(self.train_file_list)
        prefetched_indices = {}
        
        # Pre-fetch shift indices for all training samples
        for idx in range(num_samples):
            # Generate one shift_index per sample
            shift_index = rng.choice(mask_range, num_shift, replace=False)
            prefetched_indices[idx] = shift_index
        
        self.prefetched_shift_indices = prefetched_indices
        return prefetched_indices

    def trainLoader(self):
        # Creates training dataset indecies.
        train_set = list(range(len(self.train_file_list)))
        
        # Pre-fetch shift indices if enabled
        if self.use_prefetched_shift_indices:
            self.prefetch_shift_indices()

        standard_collate = make_collate(self, standard_param_queue)
        rollout_collate = make_collate(self, rollout_param_queue)

        self.train_standard_collate = standard_collate
        self.train_rollout_collate = rollout_collate

        # Create generator for deterministic shuffling
        generator = None
        manual_seed = getattr(self.global_cfg, 'manual_seed', None)
        if manual_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(manual_seed))


        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=standard_collate.standard,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=True,  # recommended for speed; safe with Manager proxy
            prefetch_factor=1,               # important: one batch prefetched per worker
            generator=generator,
        )

        # Create separate generator for rollout loader
        rollout_generator = None
        if manual_seed is not None:
            rollout_generator = torch.Generator()
            rollout_generator.manual_seed(int(manual_seed) + 1)  # Different seed to avoid same order
        
        
        rollout_batch_size = max(
            1, int(getattr(self, 'rollout_batch_size', 1)))
        self.train_rollout_data_loader = DataLoader(
            train_set,
            batch_size=rollout_batch_size,
            collate_fn=rollout_collate.rollout,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=True,
            prefetch_factor=1,
            generator=rollout_generator,
        )

    def testLoader(self):
        # Creates test dataset indecies.
        self._eval_file_list = self.test_file_list
        test_set = list(range(len(self.test_file_list)))

        # Initializes the test data loader with the specified parameters and custom collate function. Note that collate_fn is a custom function (self.testMerge) to merge and preprocess data for each batch.
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None,
                                           drop_last=False, pin_memory=False,
                                           worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        # Creates validation dataset indices based on a pre-filtered file list.
        self._eval_file_list = self.validation_file_list
        val_set = list(range(len(self.validation_file_list)))

        if len(val_set) == 0:
            self.val_data_loader = None
            return

        self.val_data_loader = DataLoader(val_set, batch_size=1, collate_fn=self.testMerge,
                                          num_workers=self.dataset_workers,
                                          shuffle=False, sampler=None,
                                          drop_last=False, pin_memory=False,
                                          worker_init_fn=self._worker_init_fn_)

    def generate_pseudo_anomaly_original(self, points, normals, center, distance_to_move=0.08):

        # print(f"distance_to_move: {distance_to_move}")

        # Find distance of each point to the center
        distances_to_center = np.linalg.norm(points - center, axis=1)

        # Find maximum distance of all points to the center
        max_distance = np.max(distances_to_center)

        # Finds a ratio for each point based on its distance to the center. Points closer to the center will have a higher ratio, meaning they will move more.
        movement_ratios = 1 - (distances_to_center / max_distance)

        # Normalizes the movement ratios to be between 0 and 1
        movement_ratios = (movement_ratios - np.min(movement_ratios)) / \
            (np.max(movement_ratios) - np.min(movement_ratios))

        # Randomly assigns a direction (inward or outward) for each point to move
        directions = np.ones(points.shape[0]) * np.random.choice([-1, 1])

        # Calculates the actual movement for each point
        movements = movement_ratios * distance_to_move * directions

        # Moves the points along their normals by the calculated movements
        new_points = points + np.abs(normals) * movements[:, np.newaxis]

        return new_points

    # -------- Helpers --------

    def _kernel(self, t, kind="cosine", q=4.0, sigma=0.35):
        """t is normalized distance; returns falloff in [0,1]."""
        t = np.clip(t, 0.0, None)
        if kind == "cosine":                    # smooth spherical cap
            x = np.clip(t, 0.0, 1.0)
            return 0.5 * (1 + np.cos(np.pi * x))
        if kind == "gaussian":                  # compact-ish, smooth
            return np.exp(-(t**2) / (2 * (sigma**2)))
        if kind == "poly":                      # (1 - t^q)+
            return np.clip(1.0 - t**q, 0.0, 1.0)
        if kind == "hard":                      # hard support
            return (t < 1.0).astype(t.dtype if hasattr(t, "dtype") else np.float32)
        raise ValueError(f"Unknown kernel: {kind}")

    def _local_frame(self, points, center, k=64):
        """PCA frame around center: columns ~ (tangent_u, tangent_v, normal)."""
        d = np.linalg.norm(points - center, axis=1)
        idx = np.argsort(d)[:k]
        Q = points[idx] - points[idx].mean(0)
        C = Q.T @ Q / max(len(idx)-1, 1)
        w, V = np.linalg.eigh(C)
        V = V[:, np.argsort(w)[::-1]]   # sort desc
        return V  # shape (3,3)

    def _beta_sample(self, rng, a, b):
        return rng.beta(a, b)

    def _choose(self, rng, items, probs=None):
        return items[rng.choice(len(items), p=np.array(probs) / sum(probs))]

    def sample_anomlay_cfg(self, rng: np.random.Generator, global_cfg) -> SmartAnomaly_Cfg:
        # Discrete choices (minimal action set)
        # print(global_cfg.cosine_kernel_prob, global_cfg.gaussian_kernel_prob, global_cfg.poly_kernel_prob, global_cfg.hard_kernel_prob)
        kernel = self._choose(rng, ["cosine", "gaussian", "poly", "hard"], probs=[
                              global_cfg.cosine_kernel_prob, global_cfg.gaussian_kernel_prob, global_cfg.poly_kernel_prob, global_cfg.hard_kernel_prob])
        dir_mode = self._choose(rng, ["normal_point", "normal_mean", "tangent_u", "tangent_v"], probs=[
                                1, 0, 0, 0])
        alpha = 1 if rng.random() < 0.5 else -1
        # Smoothing: mostly 0, sometimes 1, rarely 2
        smooth_steps = rng.choice([0, 1, 2], p=[0.7, 0.25, 0.05])

        # Continuous (Beta → range)
        # R: fraction of diameter in ????
        u_R = self._beta_sample(rng, global_cfg.R_alpha, global_cfg.R_beta)
        R_frac = global_cfg.R_low_bound + \
            (global_cfg.R_up_bound - global_cfg.R_low_bound) * u_R
        # beta (strength) in ????
        u_B = self._beta_sample(rng, global_cfg.B_alpha, global_cfg.B_beta)
        beta = global_cfg.B_low_bound + \
            (global_cfg.B_up_bound - global_cfg.B_low_bound) * u_B
        # gaussian sigma in [0.20, 0.60]; used only if kernel=="gaussian"
        u_S = self._beta_sample(rng, 1, 1)
        sigma = 0.20 + (0.60 - 0.20) * u_S
        # anisotropy e in [0.7, 1.8] → radii (e, 1/e, 1)
        u_E = self._beta_sample(rng, 1, 1)
        e = 0.70 + (1.80 - 0.70) * u_E
        radii = (float(e), float(1.0 / e), 1.0)

        # One side or double side
        one_sided = True if rng.random() < global_cfg.one_sided_prob else False

        # carve strength in [0, 0.4] (mostly small)
        u_C = self._beta_sample(rng, 1, 12)
        carve_strength = 0.0 + 0.40 * u_C
        # smoothing lambda in [0.10, 0.25]
        smooth_lambda = rng.uniform(0.10, 0.25)

        return SmartAnomaly_Cfg(
            R=R_frac,                    # we’ll set absolute R from diameter below
            beta=float(beta),
            alpha=int(alpha),
            p_bulge=0.5,
            kernel=kernel,
            q=global_cfg.poly_q,
            sigma=float(sigma),
            radii=radii,
            dir_mode=dir_mode,
            carve_strength=float(carve_strength),
            smooth_steps=int(smooth_steps),
            smooth_lambda=float(smooth_lambda),
            seed=int(rng.integers(0, 2**31 - 1)),
            one_sided=one_sided
        ), R_frac

    def _one_side_gate(self, P, center, n_hat, offset=0.0, sharpness=30.0, soft=True):
        """Global half-space gate: keep points with (P-center)·n_hat - offset > 0."""
        n_hat = np.asarray(n_hat, dtype=np.float32)
        n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-8)
        s = (P - center) @ n_hat - offset
        if soft:
            return 1.0 / (1.0 + np.exp(-sharpness * s))
        else:
            return (s > 0.0).astype(np.float32)

    def _normal_alignment_gate(self, point_normals, push_dir, cos_thresh=0.0, sharpness=30.0, soft=True):
        """Keep points whose normals align with push_dir (front-facing)."""
        push_dir = np.asarray(push_dir, dtype=np.float32)
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
        cosang = (point_normals @ push_dir)
        if soft:
            return 1.0 / (1.0 + np.exp(-sharpness * (cosang - cos_thresh)))
        else:
            return (cosang >= cos_thresh).astype(np.float32)

    # -------- Smart synthesizer (drop-in) --------

    def generate_pseudo_anomaly(self, points, normals, center, distance_to_move=0.08,
                                anomlay_cfg=None):
        """
        Upgraded version of your function. Pass `anomlay_cfg` to enable smart behavior.
        If `anomlay_cfg` is None, it behaves almost like your original (cosine cap, spherical).
        """
        # --- Defaults to preserve your old behavior ---
        if anomlay_cfg is None:
            anomlay_cfg = SmartAnomaly_Cfg(beta=distance_to_move)

        # TODO: Hard coded
        rng = np.random.default_rng()

        P = points.astype(np.float32, copy=False)
        N = normals.astype(np.float32, copy=False)
        c = center.astype(np.float32, copy=False)

        # Normalize normals softly; (your code used abs(normals) which kills direction)
        nrm = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        N = N / nrm

        # Determine radius
        diam = float(np.linalg.norm(P.max(0) - P.min(0)))
        R = anomlay_cfg.R if anomlay_cfg.R is not None else 0.2 * diam

        # Local PCA frame for anisotropy & tangents
        U = self._local_frame(P, c)  # columns: u, v, (approx) n
        ru, rv, rn = anomlay_cfg.radii

        # Coordinates in local frame and anisotropic distance
        X = (P - c) @ U         # (N,3)
        # Mahalanobis-like norm: t=1 on the ellipsoid surface
        invQ = np.diag([1.0/((ru*R)+1e-12)**2,
                        1.0/((rv*R)+1e-12)**2,
                        1.0/((rn*R)+1e-12)**2])
        t = np.sqrt(np.sum((X @ invQ) * X, axis=1))

        # Falloff weights
        w = self._kernel(t, anomlay_cfg.kernel,
                         anomlay_cfg.q, anomlay_cfg.sigma)

        if anomlay_cfg.one_sided:
            if anomlay_cfg.gate_mode == "global":
                g = self._one_side_gate(P, c, anomlay_cfg.n_global, offset=anomlay_cfg.gate_offset,
                                        sharpness=anomlay_cfg.gate_sharpness, soft=anomlay_cfg.gate_soft)
            elif anomlay_cfg.gate_mode == "normals":
                # Use mean normal direction from PCA frame as nominal outward direction
                nominal = U[:, 2]
                g = self._normal_alignment_gate(N, nominal, cos_thresh=0.0,
                                                sharpness=anomlay_cfg.gate_sharpness, soft=anomlay_cfg.gate_soft)
            else:
                raise ValueError(
                    "gate_mode must be one of {'global','normals'}")
            w = w * g  # gate the influence

        # Direction field
        if anomlay_cfg.dir_mode == "normal_point":
            D = N
        elif anomlay_cfg.dir_mode == "normal_mean":
            D = np.repeat(U[:, 2][None, :], len(P), axis=0)
        elif anomlay_cfg.dir_mode == "tangent_u":
            D = np.repeat(U[:, 0][None, :], len(P), axis=0)
        elif anomlay_cfg.dir_mode == "tangent_v":
            D = np.repeat(U[:, 1][None, :], len(P), axis=0)
        else:
            raise ValueError(f"Unknown dir_mode: {anomlay_cfg.dir_mode}")

        # Alpha (+1/-1)
        alpha = anomlay_cfg.alpha
        if alpha is None:
            alpha = 1 if rng.random() < anomlay_cfg.p_bulge else -1

        # Magnitude
        beta = anomlay_cfg.beta if anomlay_cfg.beta is not None else distance_to_move
        disp = (alpha * beta * w)[:, None] * D
        new_points = P + disp

        # # Optional carving (hole)
        # keep_mask = np.ones(len(P), dtype=bool)
        # if anomlay_cfg.carve_strength > 0.0:
        #     # remove with prob rising toward center using (1 - t)^2 inside support
        #     pr = anomlay_cfg.carve_strength * np.clip(1.0 - np.clip(t, 0, 1), 0, 1)**2
        #     keep_mask = rng.random(len(P)) > pr
        #     new_points = new_points[keep_mask]

        # # Optional light Laplacian smoothing on the deformed region
        # if anomlay_cfg.smooth_steps > 0:
        #     sub_idx = np.where(w[keep_mask] > 0.01)[0]
        #     if len(sub_idx) >= 8:
        #         sub_pts = new_points[sub_idx]
        #         k = min(16, len(sub_pts))
        #         kdt = cKDTree(sub_pts)
        #         _, nn = kdt.query(sub_pts, k=k)
        #         Xs = sub_pts.copy()
        #         for _ in range(anomlay_cfg.smooth_steps):
        #             nbr_mean = Xs[nn].mean(axis=1)
        #             Xs = Xs + anomlay_cfg.smooth_lambda * (nbr_mean - Xs)
        #         new_points[sub_idx] = Xs

        return new_points

    def testMerge(self, id, N_fixed: int = 2048, generator: torch.Generator = None):
        file_list = getattr(self, '_eval_file_list', self.test_file_list)
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original_per_sample = []     # keep per-sample for BN3
        xyz_original_cat = []            # old flat concat for backward-compat
        v2p_index_batch = []
        labels = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]

        # Optional RNG for reproducibility
        if generator is None:
            generator = torch.Generator()
            # you can set a seed here if you want determinism:
            # generator.manual_seed(0)

        for i, idx in enumerate(id):
            fn_path = file_list[idx]
            file_name.append(fn_path)

            coord, label = self._load_test_point_cloud(fn_path)

            # ---- Data aug
            Point_dict = {'coord': coord}
            Point_dict = self.test_aug_compose(Point_dict)

            # ---- numpy -> float32
            xyz = Point_dict['coord'].astype(np.float32)

            # ---- Quantize for ME sparse path (unchanged)
            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(
                xyz, xyz,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True
            )

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num += index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            # ---- Accumulate for ME batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_t = torch.from_numpy(xyz)          # (Ni, 3)
            xyz_original_per_sample.append(xyz_t)  # save per-sample
            xyz_original_cat.append(xyz_t)         # legacy flat concat
            v2p_index_batch.append(v2p_index)

            labels.append(label)

        # ---- Collate ME sparse (unchanged)
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original_cat, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))

        # =======================
        # Build BN3 tensors here
        # =======================
        B = len(xyz_original_per_sample)
        device = xyz_original.device  # keep on CPU by default; move later if you prefer
        bn3_list = []
        # indices into each sample’s original points (before BN3 sampling)
        bn3_indices = []
        orig_lengths = []

        for i in range(B):
            pts = xyz_original_per_sample[i].to(device)              # (Ni, 3)
            Ni = pts.shape[0]
            orig_lengths.append(Ni)

            if Ni >= N_fixed:
                # random sample without replacement
                idx_sel = torch.randperm(
                    Ni, generator=generator, device=pts.device)[:N_fixed]
                sampled = pts[idx_sel]
            else:
                # repeat points to reach N_fixed (keeps all points, unbiased random fill)
                num_full = N_fixed // Ni
                rem = N_fixed - num_full * Ni
                base = torch.arange(Ni, device=pts.device)
                if rem > 0:
                    extra = torch.randint(low=0, high=Ni, size=(
                        rem,), generator=generator, device=pts.device)
                    idx_sel = torch.cat([base.repeat(num_full), extra], dim=0)
                else:
                    idx_sel = base.repeat(num_full)
                sampled = pts[idx_sel]

            bn3_list.append(sampled.unsqueeze(0))  # (1, N_fixed, 3)
            bn3_indices.append(idx_sel)            # (N_fixed,)

        xyz_bn3 = torch.cat(bn3_list, dim=0).to(
            torch.float32)       # (B, N_fixed, 3)
        bn3_indices = torch.stack(
            bn3_indices, dim=0)                # (B, N_fixed)
        orig_lengths = torch.tensor(orig_lengths, dtype=torch.int32)  # (B,)

        # You can optionally move BN3 to GPU here:
        # xyz_bn3 = xyz_bn3.cuda(non_blocking=True)
        # bn3_indices = bn3_indices.cuda(non_blocking=True)
        # orig_lengths = orig_lengths.cuda(non_blocking=True)

        return {
            # existing outputs (unchanged)
            'xyz_voxel': xyz_voxel_batch,
            'feat_voxel': feat_voxel_batch,
            'xyz_original': xyz_original,         # flat concat (legacy)
            'fn': file_name,
            'v2p_index': v2p_index_batch,
            'labels': labels,
            'batch_count': batch_count,

            # new fixed-size per-sample outputs
            'xyz_bn3': xyz_bn3,                   # (B, N_fixed, 3)
            # (B, N_fixed) indices into per-sample original points
            'bn3_indices': bn3_indices,
            'orig_lengths': orig_lengths,         # (B,)
            'N_fixed': int(N_fixed),
        }
