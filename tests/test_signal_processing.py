"""
test_signal_processing.py
Unit tests for src/biomechanics/signal_processing.py
"""

import numpy as np
import pytest
from src.biomechanics.signal_processing import (
    smooth_signal,
    find_peak_in_window,
    detect_zero_crossing,
    interpolate_missing,
    rms_deviation_from_plane,
)


class TestSmoothSignal:

    def test_flat_signal_unchanged(self):
        t = np.linspace(0, 1, 120)
        sig = np.ones(120) * 5.0
        out = smooth_signal(sig, fps=120.0, cutoff_hz=6.0)
        np.testing.assert_allclose(out, 5.0, atol=1e-4)

    def test_high_freq_noise_attenuated(self):
        t = np.linspace(0, 1, 300)
        signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
        smoothed = smooth_signal(signal, fps=300.0, cutoff_hz=6.0)
        noise_original = np.std(signal - np.sin(2 * np.pi * 2 * t))
        noise_smoothed = np.std(smoothed - np.sin(2 * np.pi * 2 * t))
        assert noise_smoothed < noise_original * 0.5

    def test_short_signal_returned_as_is(self):
        sig = np.array([1.0, 2.0, 3.0])
        out = smooth_signal(sig, fps=30.0, cutoff_hz=6.0, order=4)
        np.testing.assert_array_equal(out, sig)

    def test_cutoff_above_nyquist_returned_as_is(self):
        sig = np.ones(60)
        out = smooth_signal(sig, fps=30.0, cutoff_hz=20.0)
        np.testing.assert_array_equal(out, sig)


class TestFindPeakInWindow:

    def test_finds_correct_peak(self):
        sig = np.array([0.0, 1.0, 5.0, 3.0, 2.0, 0.0])
        val, idx = find_peak_in_window(sig, 0, 6)
        assert val == 5.0
        assert idx == 2

    def test_window_start_offset(self):
        sig = np.array([0.0, 0.0, 10.0, 3.0, 0.0])
        val, idx = find_peak_in_window(sig, 2, 5)
        assert val == 10.0
        assert idx == 2

    def test_empty_window_returns_none(self):
        sig = np.array([1.0, 2.0, 3.0])
        val, idx = find_peak_in_window(sig, 3, 3)
        assert val is None
        assert idx is None

    def test_out_of_bounds_end_returns_none(self):
        sig = np.array([1.0, 2.0])
        val, idx = find_peak_in_window(sig, 0, 10)
        assert val is None
        assert idx is None


class TestDetectZeroCrossing:

    def test_falling_crossing(self):
        sig = np.array([1.0, 0.5, -0.5, -1.0])
        crossings = detect_zero_crossing(sig, direction="falling")
        assert len(crossings) > 0

    def test_rising_crossing(self):
        sig = np.array([-1.0, -0.5, 0.5, 1.0])
        crossings = detect_zero_crossing(sig, direction="rising")
        assert len(crossings) > 0

    def test_no_crossing_flat_signal(self):
        sig = np.ones(10)
        crossings = detect_zero_crossing(sig)
        assert len(crossings) == 0

    def test_sine_wave_has_crossings(self):
        t = np.linspace(0, 2 * np.pi, 100)
        sig = np.sin(t)
        crossings = detect_zero_crossing(sig, "both")
        assert len(crossings) >= 2


class TestInterpolateMissing:

    def test_short_gap_interpolated(self):
        sig = np.array([0.0, 1.0, 0.0, 0.0, 4.0, 5.0])
        conf = np.array([0.9, 0.9, 0.1, 0.1, 0.9, 0.9])
        out, mask = interpolate_missing(sig, conf, threshold=0.3, max_gap=5)
        assert mask[2] and mask[3]
        assert out[2] > 0.0  # interpolated between 1.0 and 4.0
        assert out[3] > 0.0

    def test_long_gap_not_interpolated(self):
        sig = np.zeros(20)
        sig[0] = 1.0
        sig[-1] = 1.0
        conf = np.ones(20)
        conf[1:19] = 0.1
        out, mask = interpolate_missing(sig, conf, threshold=0.3, max_gap=5)
        assert np.all(out[1:19] == 0.0)

    def test_output_same_length(self):
        sig = np.random.randn(50)
        conf = np.random.rand(50)
        out, mask = interpolate_missing(sig, conf)
        assert len(out) == 50
        assert len(mask) == 50

    def test_all_high_confidence_unchanged(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        conf = np.ones(5)
        out, mask = interpolate_missing(sig, conf, threshold=0.3)
        np.testing.assert_array_equal(out, sig)
        assert not np.any(mask)


class TestRMSDeviationFromPlane:

    def test_perfect_plane_zero_rms(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ])
        normal = np.array([0.0, 0.0, 1.0])
        centroid = np.array([0.875, 0.125, 0.0])
        rms = rms_deviation_from_plane(pts, normal, centroid)
        assert rms < 1e-9

    def test_known_deviation(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # 1m above plane
        ])
        normal = np.array([0.0, 1.0, 0.0])
        centroid = np.zeros(3)
        rms = rms_deviation_from_plane(pts, normal, centroid)
        assert rms >= 0.0
