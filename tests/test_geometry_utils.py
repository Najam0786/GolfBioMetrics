"""
test_geometry_utils.py
Unit tests for src/biomechanics/geometry_utils.py

Tests use known geometric ground truths — every assertion can be verified
by hand with basic trigonometry.
"""

import numpy as np
import pytest
from src.biomechanics.geometry_utils import (
    normalize,
    angle_between_vectors,
    signed_angle_3d,
    signed_angle_2d,
    project_onto_plane,
    fit_plane_svd,
    mirror_sagittal,
    compute_joint_angle,
    angular_velocity,
    segment_axis_horizontal,
    centre_of_mass_approx,
    point_to_plane_distance,
)


class TestNormalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-9)

    def test_non_unit_vector_normalized(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-9

    def test_zero_vector_returns_zeros(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_negative_vector(self):
        v = np.array([-2.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-9)


class TestAngleBetweenVectors:
    def test_perpendicular_returns_90(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert abs(angle_between_vectors(v1, v2) - 90.0) < 1e-6

    def test_parallel_returns_0(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([3.0, 0.0, 0.0])
        assert abs(angle_between_vectors(v1, v2) - 0.0) < 1e-6

    def test_antiparallel_returns_180(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert abs(angle_between_vectors(v1, v2) - 180.0) < 1e-6

    def test_45_degree_angle(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])
        assert abs(angle_between_vectors(v1, v2) - 45.0) < 1e-5

    def test_nearly_parallel_no_nan(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0 + 1e-10, 0.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert not np.isnan(result)
        assert 0.0 <= result <= 180.0

    def test_zero_vector_no_crash(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        result = angle_between_vectors(v1, v2)
        assert not np.isnan(result)
        assert 0.0 <= result <= 180.0

    def test_result_always_in_0_to_180(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            v1 = rng.uniform(-5, 5, 3)
            v2 = rng.uniform(-5, 5, 3)
            result = angle_between_vectors(v1, v2)
            assert 0.0 <= result <= 180.0, f"Out of range: {result}"


class TestSignedAngle3D:
    def test_positive_rotation(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        angle = signed_angle_3d(v1, v2, normal)
        assert abs(angle - 90.0) < 1e-5

    def test_negative_rotation(self):
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        angle = signed_angle_3d(v1, v2, normal)
        assert abs(angle - (-90.0)) < 1e-5

    def test_zero_angle(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        angle = signed_angle_3d(v1, v2, normal)
        assert abs(angle) < 1e-5


class TestSignedAngle2D:
    def test_90_deg_ccw(self):
        v1 = np.array([0.0, -1.0])  # pointing in -Z (golf ball direction)
        v2 = np.array([1.0,  0.0])  # pointing in +X
        angle = signed_angle_2d(v1, v2)
        assert abs(angle - 90.0) < 1e-5

    def test_3d_vectors_use_xz(self):
        v1 = np.array([1.0, 999.0, 0.0])
        v2 = np.array([0.0, 999.0, -1.0])
        angle = signed_angle_2d(v1, v2)
        assert not np.isnan(angle)

    def test_zero_angle(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([2.0, 0.0])
        assert abs(signed_angle_2d(v1, v2)) < 1e-5


class TestProjectOntoPlane:
    def test_vertical_vector_projected_to_zero_on_horizontal_plane(self):
        v = np.array([0.0, 5.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        result = project_onto_plane(v, normal)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-9)

    def test_horizontal_vector_unchanged_by_horizontal_projection(self):
        v = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        result = project_onto_plane(v, normal)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-9)

    def test_mixed_vector_reduced(self):
        v = np.array([1.0, 1.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        result = project_onto_plane(v, normal)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-9)


class TestFitPlaneSVD:
    def test_xy_plane_points_return_z_normal(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        normal, centroid = fit_plane_svd(points)
        assert abs(abs(normal[2]) - 1.0) < 1e-5

    def test_centroid_correct(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
        ])
        _, centroid = fit_plane_svd(points)
        np.testing.assert_allclose(centroid, [1.0, 2/3, 0.0], atol=1e-6)

    def test_fewer_than_3_points_no_crash(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        normal, centroid = fit_plane_svd(points)
        assert not np.any(np.isnan(normal))


class TestMirrorSagittal:
    def test_x_component_negated(self):
        kps = np.array([[1.0, 2.0, 3.0], [-0.5, 1.0, 0.5]])
        result = mirror_sagittal(kps)
        assert result[0, 0] == -1.0
        assert result[1, 0] == 0.5

    def test_y_z_unchanged(self):
        kps = np.array([[1.0, 2.0, 3.0]])
        result = mirror_sagittal(kps)
        assert result[0, 1] == 2.0
        assert result[0, 2] == 3.0


class TestComputeJointAngle:
    def test_straight_arm_180_degrees(self):
        shoulder = np.array([0.0, 1.0, 0.0])
        elbow    = np.array([0.0, 0.5, 0.0])
        wrist    = np.array([0.0, 0.0, 0.0])
        angle = compute_joint_angle(shoulder, elbow, wrist)
        assert abs(angle - 180.0) < 1e-5

    def test_right_angle_at_elbow(self):
        shoulder = np.array([0.0, 1.0, 0.0])
        elbow    = np.array([0.0, 0.0, 0.0])
        wrist    = np.array([1.0, 0.0, 0.0])
        angle = compute_joint_angle(shoulder, elbow, wrist)
        assert abs(angle - 90.0) < 1e-5


class TestAngularVelocity:
    def test_constant_angle_zero_velocity(self):
        t = np.linspace(0, 1, 60)
        angles = np.ones(60) * 45.0
        vel = angular_velocity(angles, t)
        np.testing.assert_allclose(vel, 0.0, atol=1e-6)

    def test_linear_increase_constant_velocity(self):
        t = np.linspace(0, 1, 100)
        angles = 90.0 * t
        vel = angular_velocity(angles, t)
        np.testing.assert_allclose(vel, 90.0, atol=1e-4)


class TestCentreOfMass:
    def test_mean_of_simple_points(self):
        kps = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        com = centre_of_mass_approx(kps)
        np.testing.assert_allclose(com, [1.0, 2/3, 0.0], atol=1e-9)


class TestPointToPlaneDistance:
    def test_point_on_plane_zero_distance(self):
        point  = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        plane_pt = np.array([0.0, 0.0, 0.0])
        dist = point_to_plane_distance(point, normal, plane_pt)
        assert abs(dist) < 1e-9

    def test_point_above_plane_positive(self):
        point  = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 1.0, 0.0])
        plane_pt = np.array([0.0, 0.0, 0.0])
        dist = point_to_plane_distance(point, normal, plane_pt)
        assert abs(dist - 1.0) < 1e-9
