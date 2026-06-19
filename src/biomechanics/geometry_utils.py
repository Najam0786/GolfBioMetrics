"""
geometry_utils.py
Core 3D geometry functions for golf swing biomechanics.

Design rules:
  - Pure functions: no side effects, no IO
  - Numerically stable: np.clip before arccos, atan2 for signed angles
  - Always add 1e-8 to norm denominators to prevent division by zero
  - All angle outputs in DEGREES unless explicitly noted
"""

import numpy as np


# ── Coordinate system convention ──────────────────────────────────────────────
# X: mediolateral (positive = left)
# Y: vertical     (positive = up)
# Z: anterior-posterior (positive = backward, golfer faces -Z)
# Right-handed golfer: back foot = right, front foot = left
# ─────────────────────────────────────────────────────────────────────────────


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Returns unit vector of v. Safe against zero-length vectors.

    Args:
        v: 1D array of any length

    Returns:
        unit vector (same shape), or zero vector if input magnitude < 1e-8
    """
    mag = np.linalg.norm(v)
    if mag < 1e-8:
        return np.zeros_like(v, dtype=float)
    return v / mag


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Unsigned angle between two 3D vectors in degrees [0, 180].

    Args:
        v1, v2: 3D vectors (any magnitude)

    Returns:
        angle in degrees [0.0, 180.0]

    Note:
        np.clip is mandatory — floating point errors push dot outside [-1, 1]
        causing arccos to return NaN.
    """
    n1 = normalize(v1)
    n2 = normalize(v2)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def signed_angle_3d(v1: np.ndarray, v2: np.ndarray,
                    normal: np.ndarray) -> float:
    """
    Signed angle from v1 to v2 around the given normal axis, in degrees.

    Positive = counter-clockwise when viewed from the normal direction.

    Args:
        v1, v2:  3D vectors (any magnitude)
        normal:  reference axis vector defining positive rotation direction

    Returns:
        signed angle in degrees (-180, 180]
    """
    n1 = normalize(v1)
    n2 = normalize(v2)
    cross = np.cross(n1, n2)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.degrees(np.arctan2(np.linalg.norm(cross), dot))
    sign = np.sign(np.dot(cross, normalize(normal)))
    if sign == 0:
        sign = 1.0
    return float(sign * angle)


def signed_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Signed angle from v1 to v2 in the horizontal (transverse) plane.

    Operates on the (x, z) components of 3D vectors, or directly on 2D vectors.
    Positive = counter-clockwise when viewed from above.

    Args:
        v1, v2: 2D [x, z] or 3D [x, y, z] vectors

    Returns:
        signed angle in degrees (-180, 180]
    """
    if len(v1) == 3:
        a = np.array([v1[0], v1[2]])
        b = np.array([v2[0], v2[2]])
    else:
        a = np.array(v1[:2], dtype=float)
        b = np.array(v2[:2], dtype=float)

    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.dot(a, b))
    return float(np.degrees(np.arctan2(cross, dot)))


def project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Projects vector v onto the plane defined by plane_normal.

    Used for computing transverse/frontal plane angles.

    Args:
        v:            3D vector to project
        plane_normal: unit normal of the target plane

    Returns:
        projected vector (3D, in-plane component of v)
    """
    n = normalize(plane_normal)
    return v - np.dot(v, n) * n


def fit_plane_svd(points: np.ndarray) -> tuple:
    """
    Fits a best-fit plane through a set of 3D points using SVD.

    Args:
        points: shape (N, 3), N >= 3

    Returns:
        (normal_vector, centroid): both shape (3,)

    Use case: club path consistency metric.
    """
    if len(points) < 3:
        return np.array([0.0, 1.0, 0.0]), np.mean(points, axis=0)

    centroid = np.mean(points, axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred)
    normal = vh[-1]
    return normal, centroid


def mirror_sagittal(keypoints: np.ndarray) -> np.ndarray:
    """
    Mirrors keypoints across the sagittal (YZ) plane.

    Converts right-handed to left-handed golfer representation by negating X.

    Args:
        keypoints: shape (N, 3)

    Returns:
        mirrored keypoints — same shape

    Important: after mirroring, caller must also swap left/right keypoint indices.
    """
    mirrored = keypoints.copy().astype(float)
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def compute_joint_angle(proximal: np.ndarray, joint: np.ndarray,
                        distal: np.ndarray) -> float:
    """
    Computes the angle at a joint given three 3D landmark positions.

    The angle is measured between vectors (joint → proximal) and (joint → distal).

    Args:
        proximal: 3D position of the proximal segment end (e.g., shoulder)
        joint:    3D position of the joint centre (e.g., elbow)
        distal:   3D position of the distal segment end (e.g., wrist)

    Returns:
        unsigned joint angle in degrees [0, 180]
    """
    v1 = proximal - joint
    v2 = distal - joint
    return angle_between_vectors(v1, v2)


def angular_velocity(angles_deg: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Computes angular velocity (degrees/second) from a time series of angles.

    Uses central differences (np.gradient) — smooth and unbiased.

    Args:
        angles_deg: 1D array of angles in degrees
        timestamps: 1D array of corresponding timestamps in seconds

    Returns:
        angular velocity array (degrees/second), same length as input
    """
    return np.gradient(angles_deg, timestamps)


def segment_axis_horizontal(left_kp: np.ndarray,
                            right_kp: np.ndarray) -> np.ndarray:
    """
    Returns the horizontal axis vector of a body segment (pelvis or thorax).

    Projects the left→right keypoint vector onto the horizontal (XZ) plane.

    Args:
        left_kp:  3D position of left keypoint (e.g., left_hip)
        right_kp: 3D position of right keypoint (e.g., right_hip)

    Returns:
        unit vector in the horizontal plane
    """
    axis = right_kp - left_kp
    axis[1] = 0.0  # zero out vertical component
    return normalize(axis)


def centre_of_mass_approx(keypoints: np.ndarray) -> np.ndarray:
    """
    Approximates the body centre of mass as the mean of all keypoint positions.

    Args:
        keypoints: shape (N_keypoints, 3) for a single frame

    Returns:
        COM position as shape (3,)

    Note: This is a simplified approximation. A segment-weighted model
          would be more accurate but requires body-segment mass fractions.
    """
    return np.mean(keypoints, axis=0)


def point_to_plane_distance(point: np.ndarray, plane_normal: np.ndarray,
                             plane_point: np.ndarray) -> float:
    """
    Signed distance from a point to a plane.

    Args:
        point:       3D point
        plane_normal: unit normal of the plane
        plane_point: any 3D point on the plane

    Returns:
        signed distance in metres (positive = same side as normal direction)
    """
    n = normalize(plane_normal)
    return float(np.dot(point - plane_point, n))
