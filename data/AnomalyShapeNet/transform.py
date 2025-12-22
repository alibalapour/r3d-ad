"""
Transform module for AnomalyShapeNet dataset preprocessing.

This module provides data augmentation transforms for point cloud preprocessing
including normalization, rotation, centering, and sphere crop masking.
"""

import numpy as np
import torch


class Compose:
    """Composes several transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class NormalizeCoord:
    """Normalize coordinates to [-1, 1] range based on bounding box."""
    
    def __call__(self, data):
        coord = data['coord']
        # Normalize to [-1, 1] based on max absolute coordinate
        max_val = np.abs(coord).max()
        if max_val > 0:
            coord = coord / max_val
        data['coord'] = coord
        return data


class CenterShift:
    """Center the point cloud at the origin."""
    
    def __init__(self, apply_z=True):
        self.apply_z = apply_z
    
    def __call__(self, data):
        coord = data['coord']
        if self.apply_z:
            # Center all three dimensions
            centroid = coord.mean(axis=0)
        else:
            # Center only x and y, keep z as is
            centroid = coord.mean(axis=0)
            centroid[2] = 0
        
        coord = coord - centroid
        data['coord'] = coord
        
        # Also center normals if they exist (though normals shouldn't be translated)
        # We keep normals as-is since they are direction vectors
        
        return data


class RandomRotate:
    """Random rotation around a specified axis."""
    
    def __init__(self, angle=[-1, 1], axis="z", center=None, p=1.0):
        """
        Args:
            angle: Range for random angle selection [min, max] or single value
            axis: Rotation axis ("x", "y", or "z")
            center: Center point for rotation (default: origin)
            p: Probability of applying rotation
        """
        if isinstance(angle, (list, tuple)):
            self.angle_range = angle
        else:
            self.angle_range = [-angle, angle]
        self.axis = axis.lower()
        self.center = center if center is not None else [0, 0, 0]
        self.p = p
    
    def __call__(self, data):
        if np.random.random() > self.p:
            return data
        
        coord = data['coord']
        normal = data.get('normal', None)
        
        # Random angle in radians
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        angle_rad = np.deg2rad(angle)
        
        # Create rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        if self.axis == 'z':
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        elif self.axis == 'y':
            rot_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif self.axis == 'x':
            rot_matrix = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        else:
            raise ValueError(f"Invalid axis: {self.axis}. Must be 'x', 'y', or 'z'.")
        
        # Apply rotation
        center = np.array(self.center)
        coord = coord - center
        coord = coord @ rot_matrix.T
        coord = coord + center
        data['coord'] = coord
        
        # Rotate normals if they exist
        if normal is not None:
            normal = normal @ rot_matrix.T
            data['normal'] = normal
        
        return data


class SphereCropMask:
    """
    Divide point cloud into spherical patches for pseudo-anomaly synthesis.
    
    This creates a mask that divides the point cloud into multiple spherical regions
    centered around randomly selected seed points. Each point is assigned to the
    nearest seed, creating non-overlapping patches.
    """
    
    def __init__(self, part_num=8):
        """
        Args:
            part_num: Number of spherical patches to create
        """
        self.part_num = part_num
    
    def __call__(self, data):
        coord = data['coord']
        mask = data.get('mask', np.ones(coord.shape[0]) * -1)
        
        # Get number of points
        num_points = coord.shape[0]
        
        # Randomly select seed points for patches
        if num_points >= self.part_num:
            # Select part_num seed points randomly
            seed_indices = np.random.choice(num_points, self.part_num, replace=False)
        else:
            # If fewer points than patches, use all points as seeds
            seed_indices = np.arange(num_points)
        
        seed_points = coord[seed_indices]
        
        # Compute distances from all points to all seeds
        # Shape: (num_points, part_num)
        distances = np.linalg.norm(
            coord[:, np.newaxis, :] - seed_points[np.newaxis, :, :],
            axis=2
        )
        
        # Assign each point to nearest seed (create mask)
        nearest_seed = np.argmin(distances, axis=1)
        
        # Update mask with patch assignments
        mask = nearest_seed.astype(np.int32)
        data['mask'] = mask
        
        # Return centers for potential use in anomaly generation
        centers = seed_points
        
        return data, centers
