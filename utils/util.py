import random
import numpy as np
import open3d as o3d
from itertools import repeat
from dataclasses import dataclass, asdict
from typing import Callable

def random_rorate(pc):
    degree = np.random.uniform(-180, 180, 3)
    matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.pi * degree / 180.0)
    pc_aug = np.matmul(pc, matrix)
    return pc_aug.astype(np.float32)

def split_pointcloud(pc, matrix):
    R = matrix[:3, :3]
    T = matrix[:3, 3]
    
    # The plane's normal vector is the z-axis of the transformed coordinate system
    normal_vector = R[:, 2]
    point_on_plane = T
    
    pc1 = []
    pc2 = []
    
    for p in pc:
        distance = np.dot(normal_vector, p - point_on_plane)
        if distance > 0:
            pc1.append(p)
        else:
            pc2.append(p)
    
    pc1 = np.array(pc1, np.float32)
    pc2 = np.array(pc2, np.float32)
    
    return pc1, pc2

def rotate_vector(vector, axis, angle_degrees):
    """
    Rotate a vector around a given axis by a specified angle.
    """
    angle_radians = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    cross_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_matrix + (1 - cos_angle) * np.outer(axis, axis)
    return np.dot(rotation_matrix, vector)

def trans_and_rotate_plane(matrix, distance, degree1, degree2):
    normal_vector = matrix[:3, 2]
    point_on_plane = matrix[:3, 3]
    
    # Move the plane along its normal direction
    point_on_plane += normal_vector * distance
    
    # Identify two orthogonal axes for rotation
    # For simplicity, assuming these axes are the x and y axes of the local plane coordinate system
    axis1 = matrix[:3, 0]  # Local x-axis
    axis2 = matrix[:3, 1]  # Local y-axis
    
    normal_vector = rotate_vector(normal_vector, axis1, degree1)
    normal_vector = rotate_vector(normal_vector, axis2, degree2)
    
    new_matrix = np.eye(4)
    new_matrix[:3, 2] = normal_vector  # Update the normal vector
    new_matrix[:3, 3] = point_on_plane  # Update the point on plane
    
    return new_matrix

def random_split(pc, matrix, distance=0.5, degree=5):
    distance = np.random.uniform(-distance, distance)
    degree1 = np.random.uniform(-degree, degree)
    degree2 = np.random.uniform(-degree, degree)
    new_matrix = trans_and_rotate_plane(matrix, distance, degree1, degree2)
    pc1, pc2 = split_pointcloud(pc, new_matrix)
    return pc1, pc2

def distance_square(p1, p2):
    tensor = p1 - p2
    val = tensor.mul(tensor).sum()
    return val

def normalize(pc):
    pc -= np.mean(pc, axis=0)
    pc /= np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc.astype(np.float32)

def normalize_vector(vector):
    length = np.linalg.norm(vector)
    unit_vector = vector / length
    return unit_vector

def random_patch(points, patch_num, scale):
    patchP = np.zeros((patch_num, 3), dtype=np.float32)

    view = np.array([[2, -2, -2], [2, -2, 0], [2, -2, 2], [0, -2, -2], [0, -2, 0], [0, -2, 2],
                     [-2, -2, -2], [-2, -2, 0], [-2, -2, 2], [2, 0, -2], [2, 0, 0], [2, 0, 2],
                     [0, 0, -2], [0, 0, 2], [-2, 0, -2], [-2, 0, 0], [-2, 0, 2], [2, 2, -2],
                     [2, 2, 0], [2, 2, 2], [0, 2, -2], [0, 2, 0], [0, 2, 2], [-2, 2, -2],
                     [-2, 2, 0], [-2, 2, 2]], dtype=np.float32)
    
    index = view[random.randint(0, view.shape[0] - 1)][None, :]
    
    # Calculate distances from points to the selected view point
    distances = np.sum((points - index) ** 2, axis=1)
    distance_order = distances.argsort()
    
    for sp in range(patch_num):
        patchP[sp] = points[distance_order[sp]]
    
    # Calculate normal and apply transformations
    convex_normal = patchP-index
    convex_normal = np.apply_along_axis(normalize_vector, 1, convex_normal)
    convex_normal *= scale if random.random() < 0.5 else -scale

    # Generate a random distance for each point that follows a Gaussian distribution
    random_translation_distances =np.random.rand(patch_num)
    sort_random_translation_distances = np.sort(random_translation_distances)[::-1]

    # Generate random translation vector
    random_translation_vectors = convex_normal * sort_random_translation_distances[:, None]
    patchP += random_translation_vectors
    
    # Update original points with the transformed patch
    new_points = points.copy()
    new_points[distance_order[:patch_num]] = patchP
    
    mask = np.zeros(points.shape[0], np.float32)
    mask[distance_order[:patch_num]] = 1
    
    return new_points, mask

def random_translate(points, radius=0.08, translate=0.02, part=16):
    mask = np.zeros(points.shape[0], np.float32)

    for _ in range(part):
        random_point = points[random.randint(0, points.shape[0] - 1)]
        distances = np.linalg.norm(points - random_point, axis=1)
        fg = distances < radius

        point_fg = points[fg]
        n, dim= point_fg.shape
        t = list(repeat(translate, times=dim))

        ts = []
        for d in range(dim):
            ts.append(np.random.uniform(-abs(t[d]), abs(t[d]), n))
        
        points[fg] += np.stack(ts, axis=-1)
        mask[fg] = 1

    return points, mask


# ============================================================================
# Pseudo Anomaly Synthesis Module
# ============================================================================

def _rand_sign(p_plus=0.5):
    """+1 with prob p_plus, else -1."""
    return +1 if np.random.rand() < float(p_plus) else -1


@dataclass
class SmartAnomaly_Cfg:
    """Configuration for smart pseudo anomaly synthesis."""
    # size & strength
    R: float = None            # support radius; None -> 0.2 * object diameter
    beta: float = 0.08         # magnitude (your distance_to_move)
    alpha: int = None          # +1 bulge, -1 cavity, None -> random w.p. p_bulge
    p_bulge: float = 0.5

    # falloff kernel
    kernel: str = "cosine"     # {"cosine","gaussian","poly","hard"}
    q: float = 2.0
    sigma: float = 0.35        # as fraction of R for gaussian

    # anisotropy (ellipsoid radii along local frame axes)
    radii: tuple = (1.0, 1.0, 1.0)    # (ru, rv, rn); 1,1,1 == sphere

    # displacement direction
    # {"normal_point","normal_mean","tangent_u","tangent_v"}
    dir_mode: str = "normal_point"

    # gating (NEW): keep deformation on one side only
    one_sided: bool = True              # turn on to gate to one side of a plane
    gate_mode: str = "normals"          # {"global","normals"}
    # global outward direction (used when gate_mode="global")
    n_global: tuple = (0.0, 0.0, 1.0)
    gate_soft: bool = True              # soft logistic gate vs hard step
    gate_sharpness: float = 30.0        # larger = crisper
    # shift plane along n (push gate outward)
    gate_offset: float = 0.0

    # extras
    carve_strength: float = 0.0         # 0..1 probability of removing center points (holes)
    smooth_steps: int = 0               # Laplacian smoothing steps inside support
    smooth_lambda: float = 0.15
    seed: int = None


class AnomalyPreset:
    """
    Factory for anomaly configs. Reads parameter ranges from `args`.

    Expected args fields (with sensible fallbacks used via getattr):
      - R_low_bound, R_up_bound, R_alpha, R_beta
      - B_low_bound, B_up_bound, B_alpha, B_beta
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
            self.type_7_shear_u,
            self.type_7b_shear_v,
            self.type_8_double_sided_ripple,
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
    # 3) Ridge (elongated anisotropic bulge)
    # ----------------------
    def type_3_ridge(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(2.0, 0.6, 0.8),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=+1,
            beta=B,
            sigma=0.35
        )

    # ----------------------
    # 4) Trench (elongated anisotropic dent)
    # ----------------------
    def type_4_trench(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(2.0, 0.6, 0.8),
            kernel="cosine",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,
            beta=B,
            sigma=0.35
        )

    # ----------------------
    # 5) Elliptic Patch / Flat Spot (pressed/flattened region)
    # ----------------------
    def type_5_elliptic_patch_flat_spot(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.3),
            kernel="poly",
            q=4.0,
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,
            beta=B
        )

    # ----------------------
    # 6) Skewed Impact Crater (oblique one-sided dent)
    # ----------------------
    def type_6_skewed_impact_crater(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.5, 0.8, 0.6),
            kernel="gaussian",
            one_sided=True,
            gate_mode="normals",
            dir_mode="normal_mean",
            alpha=-1,
            beta=B,
            sigma=0.4
        )

    # ----------------------
    # 7) Shear along U (tangential slip deformation)
    # ----------------------
    def type_7_shear_u(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.8),
            kernel="cosine",
            dir_mode="tangent_u",
            one_sided=False,
            alpha=-1,
            beta=B
        )

    # ----------------------
    # 7b) Shear along V (tangential slip in perpendicular direction)
    # ----------------------
    def type_7b_shear_v(self):
        R, B = self.get_R_B()
        return SmartAnomaly_Cfg(
            R=R,
            radii=(1.2, 1.2, 0.8),
            kernel="cosine",
            dir_mode="tangent_v",
            one_sided=False,
            alpha=-1,
            beta=B
        )

    # ----------------------
    # 8) Double-Sided Ripple (cosine, alternating sign)
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
        beta_scaled = B * self._p("drag_beta_scale", 0.17)
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


def _kernel(t, kind="cosine", q=4.0, sigma=0.35):
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


def _local_frame(points, center, k=64):
    """PCA frame around center: columns ~ (tangent_u, tangent_v, normal)."""
    d = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(d)[:k]
    Q = points[idx] - points[idx].mean(0)
    C = Q.T @ Q / max(len(idx)-1, 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]]   # sort desc
    return V  # shape (3,3)


def _one_side_gate(P, center, n_hat, offset=0.0, sharpness=30.0, soft=True):
    """Global half-space gate: keep points with (P-center)·n_hat - offset > 0."""
    n_hat = np.asarray(n_hat, dtype=np.float32)
    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-8)
    s = (P - center) @ n_hat - offset
    if soft:
        return 1.0 / (1.0 + np.exp(-sharpness * s))
    else:
        return (s > 0.0).astype(np.float32)


def _normal_alignment_gate(point_normals, push_dir, cos_thresh=0.0, sharpness=30.0, soft=True):
    """Keep points whose normals align with push_dir (front-facing)."""
    push_dir = np.asarray(push_dir, dtype=np.float32)
    push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
    cosang = (point_normals @ push_dir)
    if soft:
        return 1.0 / (1.0 + np.exp(-sharpness * (cosang - cos_thresh)))
    else:
        return (cosang >= cos_thresh).astype(np.float32)


def pseudo_anomaly_synthesis(points, normals, center, anomaly_cfg=None):
    """
    Generate pseudo anomaly on a point cloud patch using smart anomaly synthesis.
    
    Args:
        points: numpy array of shape (N, 3) - point cloud coordinates
        normals: numpy array of shape (N, 3) - point cloud normals
        center: numpy array of shape (3,) - center of the anomaly region
        anomaly_cfg: SmartAnomaly_Cfg instance with anomaly parameters
        
    Returns:
        new_points: numpy array of shape (N, 3) - deformed point cloud
    """
    # --- Defaults to preserve old behavior ---
    if anomaly_cfg is None:
        anomaly_cfg = SmartAnomaly_Cfg(beta=0.08)

    rng = np.random.default_rng()

    P = points.astype(np.float32, copy=False)
    N = normals.astype(np.float32, copy=False)
    c = center.astype(np.float32, copy=False)

    # Normalize normals softly
    nrm = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    N = N / nrm

    # Determine radius
    diam = float(np.linalg.norm(P.max(0) - P.min(0)))
    R = anomaly_cfg.R if anomaly_cfg.R is not None else 0.2 * diam

    # Local PCA frame for anisotropy & tangents
    U = _local_frame(P, c)  # columns: u, v, (approx) n
    ru, rv, rn = anomaly_cfg.radii

    # Coordinates in local frame and anisotropic distance
    X = (P - c) @ U         # (N,3)
    # Mahalanobis-like norm: t=1 on the ellipsoid surface
    invQ = np.diag([1.0/((ru*R)+1e-12)**2,
                    1.0/((rv*R)+1e-12)**2,
                    1.0/((rn*R)+1e-12)**2])
    t = np.sqrt(np.sum((X @ invQ) * X, axis=1))

    # Falloff weights
    w = _kernel(t, anomaly_cfg.kernel, anomaly_cfg.q, anomaly_cfg.sigma)

    if anomaly_cfg.one_sided:
        if anomaly_cfg.gate_mode == "global":
            g = _one_side_gate(P, c, anomaly_cfg.n_global, offset=anomaly_cfg.gate_offset,
                              sharpness=anomaly_cfg.gate_sharpness, soft=anomaly_cfg.gate_soft)
        elif anomaly_cfg.gate_mode == "normals":
            # Use mean normal direction from PCA frame as nominal outward direction
            nominal = U[:, 2]
            g = _normal_alignment_gate(N, nominal, cos_thresh=0.0,
                                      sharpness=anomaly_cfg.gate_sharpness, soft=anomaly_cfg.gate_soft)
        else:
            raise ValueError("gate_mode must be one of {'global','normals'}")
        w = w * g  # gate the influence

    # Direction field
    if anomaly_cfg.dir_mode == "normal_point":
        D = N
    elif anomaly_cfg.dir_mode == "normal_mean":
        D = np.repeat(U[:, 2][None, :], len(P), axis=0)
    elif anomaly_cfg.dir_mode == "tangent_u":
        D = np.repeat(U[:, 0][None, :], len(P), axis=0)
    elif anomaly_cfg.dir_mode == "tangent_v":
        D = np.repeat(U[:, 1][None, :], len(P), axis=0)
    else:
        raise ValueError(f"Unknown dir_mode: {anomaly_cfg.dir_mode}")

    # Alpha (+1/-1)
    alpha = anomaly_cfg.alpha
    if alpha is None:
        alpha = 1 if rng.random() < anomaly_cfg.p_bulge else -1

    # Magnitude
    beta = anomaly_cfg.beta
    disp = (alpha * beta * w)[:, None] * D
    new_points = P + disp

    return new_points
