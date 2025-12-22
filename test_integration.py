"""
Example demonstration of pseudo anomaly synthesis integration.

This script shows how the pseudo anomaly synthesis works without requiring
a full dataset. It creates synthetic point clouds and demonstrates the
anomaly generation process.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Pseudo Anomaly Synthesis - Integration Example")
print("=" * 80)

# Test 1: Import transform module
print("\n[1/5] Testing transform module imports...")
try:
    from data.AnomalyShapeNet import transform
    print("✓ Successfully imported data.AnomalyShapeNet.transform")
    
    # Verify all required classes exist
    required_classes = ['Compose', 'NormalizeCoord', 'CenterShift', 
                       'RandomRotate', 'SphereCropMask']
    for cls_name in required_classes:
        assert hasattr(transform, cls_name), f"Missing class: {cls_name}"
        print(f"  ✓ {cls_name} class found")
    
except Exception as e:
    print(f"✗ Failed to import transform module: {e}")
    sys.exit(1)

# Test 2: Import preprocessing module
print("\n[2/5] Testing preprocessing module imports...")
try:
    from preprocessing import (
        AnomalyPreset, 
        SmartAnomaly_Cfg, 
        Dataset,
        CollateBundle
    )
    print("✓ Successfully imported preprocessing classes")
    print("  ✓ AnomalyPreset")
    print("  ✓ SmartAnomaly_Cfg")
    print("  ✓ Dataset")
    print("  ✓ CollateBundle")
    
except Exception as e:
    print(f"✗ Failed to import preprocessing module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create synthetic point cloud
print("\n[3/5] Creating synthetic point cloud...")
try:
    # Create a simple sphere point cloud
    num_points = 2048
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    normals = points.copy()  # For sphere, normals = position
    
    print(f"✓ Created sphere point cloud with {num_points} points")
    print(f"  Shape: {points.shape}")
    print(f"  Range: [{points.min():.3f}, {points.max():.3f}]")
    
except Exception as e:
    print(f"✗ Failed to create point cloud: {e}")
    sys.exit(1)

# Test 4: Test transform pipeline
print("\n[4/5] Testing transform pipeline...")
try:
    # Create transform pipeline
    normalize = transform.NormalizeCoord()
    center_shift = transform.CenterShift(apply_z=True)
    rotate_z = transform.RandomRotate(angle=[-45, 45], axis="z", p=1.0)
    sphere_crop = transform.SphereCropMask(part_num=8)
    
    # Apply transforms
    data = {'coord': points.copy(), 'normal': normals.copy()}
    
    print("  Applying NormalizeCoord...")
    data = normalize(data)
    
    print("  Applying CenterShift...")
    data = center_shift(data)
    
    print("  Applying RandomRotate...")
    data = rotate_z(data)
    
    print("  Applying SphereCropMask...")
    data, centers = sphere_crop(data)
    
    print(f"✓ Transform pipeline completed successfully")
    print(f"  Number of patches: {len(np.unique(data['mask']))}")
    print(f"  Centers shape: {centers.shape}")
    
except Exception as e:
    print(f"✗ Failed to apply transforms: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test anomaly preset configurations
print("\n[5/5] Testing anomaly preset configurations...")
try:
    from easydict import EasyDict
    
    # Create config
    cfg = EasyDict({
        'R_low_bound': 0.10,
        'R_up_bound': 0.30,
        'R_alpha': 2.0,
        'R_beta': 5.0,
        'B_low_bound': 0.02,
        'B_up_bound': 0.15,
        'B_alpha': 2.0,
        'B_beta': 5.0,
    })
    
    # Create anomaly preset factory
    preset_factory = AnomalyPreset(cfg)
    
    print(f"✓ Created AnomalyPreset with {len(preset_factory.presets)} presets:")
    preset_names = [
        "Basic Bulge",
        "Basic Dent",
        "Ridge",
        "Trench",
        "Elliptic Patch",
        "Skewed Impact Crater",
        "Shear U",
        "Shear V",
        "Double-Sided Ripple",
        "Micro Dimple Field",
        "Directional Drag"
    ]
    
    for num, name in enumerate(preset_names, 1):
        print(f"  {num}. {name}")
    
    # Test generating a preset config
    print("\n  Generating sample anomaly configs...")
    for i in range(3):
        preset_fn = preset_factory.presets[i]
        config = preset_fn()
        print(f"    Preset {i+1}: R={config.R:.3f}, beta={config.beta:.3f}, "
              f"kernel={config.kernel}, alpha={config.alpha}")
    
    print("\n✓ All anomaly presets accessible")
    
except Exception as e:
    print(f"✗ Failed to test anomaly presets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("Integration Test Summary")
print("=" * 80)
print("✓ Transform module: OK")
print("✓ Preprocessing module: OK")
print("✓ Point cloud generation: OK")
print("✓ Transform pipeline: OK")
print("✓ Anomaly presets: OK")
print("\nAll integration tests passed!")
print("\nYou can now use the pseudo anomaly synthesis module for training.")
print("See README.md and PSEUDO_ANOMALY_INTEGRATION.md for usage examples.")
print("=" * 80)
