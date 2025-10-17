import numpy as np

from amdm.amdm import AMDMConfig, AMDMMonitor


def test_amdm_low_variation_results_in_low_score():
    monitor = AMDMMonitor(["a", "b"], AMDMConfig(alpha=0.3, cov_window=10, min_baseline=3))
    steady_values = [np.array([1.0, 2.0])] * 15
    for value in steady_values:
        state = monitor.update(value)
    assert state.last_score < 1.0


def test_amdm_outlier_triggers_high_score():
    monitor = AMDMMonitor(["a", "b"], AMDMConfig(alpha=0.3, cov_window=10, min_baseline=3))
    for _ in range(12):
        monitor.update([1.0, 2.0])
    state = monitor.update([5.0, 9.0])
    assert state.last_score > 2.5


def test_amdm_state_roundtrip_preserves_statistics():
    monitor = AMDMMonitor(["a", "b"], AMDMConfig(alpha=0.2, cov_window=20, min_baseline=5))
    for value in ([1.0, 2.0], [1.5, 2.5], [2.0, 3.0]):
        monitor.update(value)
    snapshot = monitor.to_dict()
    restored = AMDMMonitor.from_dict(snapshot)
    np.testing.assert_allclose(restored._means, monitor._means)
    np.testing.assert_allclose(restored._variances, monitor._variances)
    assert restored.config == monitor.config
