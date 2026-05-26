"""Tests for model store and A/B testing."""
from __future__ import annotations

import math

import pytest
from src.registry.ab_testing import (
    ABTest,
    ABTestRouter,
    ExperimentStatus,
    Variant,
    _normal_cdf,
)
from src.registry.model_store import ModelStore, VersionStatus

# ------------------------------------------------------------------
# Model Store tests
# ------------------------------------------------------------------


class TestModelStore:
    def test_register(self) -> None:
        store = ModelStore()
        mv = store.register("dana-base", "v1", "http://localhost:9000")
        assert mv.model_name == "dana-base"
        assert mv.version == "v1"
        assert mv.status == VersionStatus.STAGING

    def test_promote(self) -> None:
        store = ModelStore()
        store.register("dana-base", "v1", "http://localhost:9000")
        mv = store.promote("dana-base", "v1")
        assert mv.status == VersionStatus.ACTIVE
        assert mv.promoted_at is not None

    def test_promote_deprecates_previous(self) -> None:
        store = ModelStore()
        store.register("dana-base", "v1", "http://localhost:9000")
        store.promote("dana-base", "v1")
        store.register("dana-base", "v2", "http://localhost:9001")
        store.promote("dana-base", "v2")

        versions = store.list_versions("dana-base")
        assert versions[0].status == VersionStatus.DEPRECATED
        assert versions[1].status == VersionStatus.ACTIVE

    def test_rollback(self) -> None:
        store = ModelStore()
        store.register("dana-base", "v1", "http://localhost:9000")
        store.promote("dana-base", "v1")
        store.register("dana-base", "v2", "http://localhost:9001")
        store.promote("dana-base", "v2")

        rolled = store.rollback("dana-base")
        assert rolled.version == "v1"
        assert rolled.status == VersionStatus.ACTIVE

        versions = store.list_versions("dana-base")
        v2 = next(v for v in versions if v.version == "v2")
        assert v2.status == VersionStatus.ROLLED_BACK

    def test_rollback_no_previous_raises(self) -> None:
        store = ModelStore()
        store.register("dana-base", "v1", "http://localhost:9000")
        store.promote("dana-base", "v1")
        with pytest.raises(ValueError, match="No previous version"):
            store.rollback("dana-base")

    def test_get_active(self) -> None:
        store = ModelStore()
        store.register("dana-base", "v1", "http://localhost:9000")
        assert store.get_active("dana-base") is None
        store.promote("dana-base", "v1")
        active = store.get_active("dana-base")
        assert active is not None
        assert active.version == "v1"

    def test_list_models(self) -> None:
        store = ModelStore()
        store.register("dana-base", "v1", "http://localhost:9000")
        store.register("dana-code", "v1", "http://localhost:9001")
        assert sorted(store.list_models()) == ["dana-base", "dana-code"]

    def test_promote_nonexistent_raises(self) -> None:
        store = ModelStore()
        with pytest.raises(ValueError, match="not found"):
            store.promote("dana-base", "v99")


# ------------------------------------------------------------------
# A/B Testing tests
# ------------------------------------------------------------------


class TestABTest:
    def _make_test(self) -> ABTest:
        return ABTest(
            experiment_id="exp-1",
            model_name="dana-base",
            variants=[
                Variant(name="control", model_version="v1", weight=0.5),
                Variant(name="treatment", model_version="v2", weight=0.5),
            ],
            confidence_level=0.95,
        )

    def test_route_returns_variant(self) -> None:
        test = self._make_test()
        variant = test.route()
        assert variant.name in ("control", "treatment")

    def test_record_outcome(self) -> None:
        test = self._make_test()
        test.record_outcome("control", success=True)
        test.record_outcome("control", success=False)
        test.record_outcome("treatment", success=True)

        assert test.variants[0].successes == 1
        assert test.variants[0].failures == 1
        assert test.variants[1].successes == 1

    def test_record_unknown_variant_raises(self) -> None:
        test = self._make_test()
        with pytest.raises(ValueError, match="Unknown variant"):
            test.record_outcome("nonexistent", success=True)

    def test_significance_insufficient_data(self) -> None:
        test = self._make_test()
        for _ in range(5):
            test.record_outcome("control", True)
            test.record_outcome("treatment", True)
        result = test.check_significance()
        assert result is None  # fewer than 10 per variant

    def test_significance_with_identical_rates(self) -> None:
        """When both variants have the same success rate, result should not be significant."""
        test = self._make_test()
        for _ in range(100):
            test.record_outcome("control", True)
            test.record_outcome("treatment", True)
        result = test.check_significance()
        assert result is not None
        assert result.significant is False
        assert result.p_value == 1.0  # identical rates => z=0

    def test_significance_with_divergent_rates(self) -> None:
        """A large difference in success rates should yield significance."""
        test = self._make_test()
        # Control: 50% success
        for _ in range(200):
            test.record_outcome("control", True)
        for _ in range(200):
            test.record_outcome("control", False)
        # Treatment: 80% success
        for _ in range(320):
            test.record_outcome("treatment", True)
        for _ in range(80):
            test.record_outcome("treatment", False)

        result = test.check_significance()
        assert result is not None
        assert result.significant is True
        assert result.p_value < 0.05
        assert abs(result.control_rate - 0.5) < 0.01
        assert abs(result.treatment_rate - 0.8) < 0.01

    def test_z_statistic_manual_verification(self) -> None:
        """Verify the Z-statistic matches a manual calculation."""
        test = self._make_test()
        # Control: 60/100 successes
        for _ in range(60):
            test.record_outcome("control", True)
        for _ in range(40):
            test.record_outcome("control", False)
        # Treatment: 75/100 successes
        for _ in range(75):
            test.record_outcome("treatment", True)
        for _ in range(25):
            test.record_outcome("treatment", False)

        result = test.check_significance()
        assert result is not None

        # Manual: p1=0.6, p2=0.75, n1=n2=100
        p_pool = (60 + 75) / (100 + 100)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / 100 + 1 / 100))
        expected_z = (0.6 - 0.75) / se

        assert abs(result.z_statistic - expected_z) < 1e-6

    def test_normal_cdf_known_values(self) -> None:
        """Sanity check the normal CDF approximation."""
        assert abs(_normal_cdf(0.0) - 0.5) < 1e-6
        assert abs(_normal_cdf(1.96) - 0.975) < 0.001
        assert abs(_normal_cdf(-1.96) - 0.025) < 0.001

    def test_experiment_becomes_significant(self) -> None:
        """Status should transition to SIGNIFICANT once threshold is crossed."""
        test = self._make_test()
        assert test.status == ExperimentStatus.RUNNING
        # Large divergence
        for _ in range(500):
            test.record_outcome("control", True)
        for _ in range(500):
            test.record_outcome("control", False)
        for _ in range(900):
            test.record_outcome("treatment", True)
        for _ in range(100):
            test.record_outcome("treatment", False)
        test.check_significance()
        assert test.status == ExperimentStatus.SIGNIFICANT


class TestABTestRouter:
    def test_create_and_retrieve(self) -> None:
        router = ABTestRouter()
        test = router.create_experiment(
            experiment_id="exp-1",
            model_name="dana-base",
            variants=[
                {"name": "control", "model_version": "v1", "weight": 0.5},
                {"name": "treatment", "model_version": "v2", "weight": 0.5},
            ],
        )
        assert test.experiment_id == "exp-1"
        retrieved = router.get_experiment("exp-1")
        assert retrieved is test

    def test_stop_experiment(self) -> None:
        router = ABTestRouter()
        router.create_experiment(
            experiment_id="exp-1",
            model_name="dana-base",
            variants=[{"name": "a", "model_version": "v1"}],
        )
        router.stop_experiment("exp-1")
        assert router.get_experiment("exp-1").status == ExperimentStatus.STOPPED

    def test_list_experiments(self) -> None:
        router = ABTestRouter()
        router.create_experiment("e1", "m1", [{"name": "a", "model_version": "v1"}])
        router.create_experiment("e2", "m2", [{"name": "b", "model_version": "v2"}])
        assert len(router.list_experiments()) == 2
