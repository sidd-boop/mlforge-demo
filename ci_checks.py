"""
User-defined CI checks, inference runner, and metric strategy.
A real user would write these to define:
  - How their model is validated (BaseCheck subclasses)
  - How inference runs (BaseInferenceRunner subclass)
  - What metric thresholds must be met (BaseMetricStrategy)
"""
import os
import numpy as np
import joblib
from src.mlforge.core.deployment_manager.ci.checks import BaseCheck
from src.mlforge.core.deployment_manager.ci.results import CheckResult
from src.mlforge.core.deployment_manager.ci.inference import (
    BaseInferenceRunner,
    BaseInferenceInput,
    BaseInferenceOutput,
)
from src.mlforge.core.deployment_manager.ci.metrics import (
    MetricContext,
    BaselineComparisonStrategy,
)
# ─── Inference Schemas ──────────────────────────────────────
class IrisInput(BaseInferenceInput):
    """Input schema: 4 features for Iris classification."""
    features: list
class IrisOutput(BaseInferenceOutput):
    """Output schema: predicted class + probabilities."""
    prediction: int
    probabilities: list
# ─── Inference Runner ───────────────────────────────────────
class IrisInferenceRunner(BaseInferenceRunner[IrisInput, IrisOutput]):
    """
    Loads a joblib-saved SGDClassifier and runs inference.
    This is what the CI framework calls via runner.run(raw_dict).
    """
    input_model = IrisInput
    output_model = IrisOutput
    def load(self, model_path: str):
        self.model = joblib.load(model_path)
    def predict(self, inp: IrisInput) -> IrisOutput:
        features = np.array(inp.features).reshape(1, -1)
        pred = int(self.model.predict(features)[0])
        probs = self.model.predict_proba(features)[0].tolist()
        return IrisOutput(prediction=pred, probabilities=probs)
    def batch(self, inputs):
        return [self.predict(inp) for inp in inputs]

# ═════════════════════════════════════════════════════════════
# HAPPY-PATH CHECKS (Phase 5A)
# ═════════════════════════════════════════════════════════════
def register_all_checks(registry):
    """
    Registers all CI checks on the given CheckRegistry instance.
    Called by run_e2e.py before building the CI pipeline.
    """
    @registry.register(name="model_file_exists", priority=0, required=True)
    class ModelFileExistsCheck(BaseCheck):
        """Check that the model file exists on disk."""
        def execute(self, context):
            path = context.model_path
            if os.path.exists(path):
                size = os.path.getsize(path)
                return CheckResult(
                    passed=True,
                    message=f"Model found at {path} ({size} bytes)",
                )
            return CheckResult(
                passed=False,
                message=f"Model NOT found at {path}",
            )
    @registry.register(
        name="model_loads_correctly",
        priority=1,
        required=True,
        depends_on=["model_file_exists"],
    )
    class ModelLoadsCheck(BaseCheck):
        """Check that the model can be deserialized without error."""
        def execute(self, context):
            try:
                model = joblib.load(context.model_path)
                context["loaded_model"] = model
                return CheckResult(
                    passed=True,
                    message=f"Model loaded successfully: {type(model).__name__}",
                )
            except Exception as e:
                return CheckResult(passed=False, message=f"Load failed: {e}")
    @registry.register(
        name="single_inference",
        priority=2,
        required=True,
        depends_on=["model_loads_correctly"],
    )
    class SingleInferenceCheck(BaseCheck):
        """
        Run a single inference through the runner.
        Tests: runner.load(), runner.run() (which calls predict + validates + logs).
        """
        def execute(self, context):
            runner = context.runner
            if runner is None:
                return CheckResult(passed=False, message="No inference runner in context")
            try:
                runner.load(context.model_path)
                sample = context.inference_sample or {"features": [5.1, 3.5, 1.4, 0.2]}
                output = runner.run(sample)
                return CheckResult(
                    passed=True,
                    message=f"Inference OK: prediction={output.prediction}",
                    details={
                        "prediction": output.prediction,
                        "probabilities": output.probabilities,
                    },
                )
            except Exception as e:
                return CheckResult(passed=False, message=f"Inference failed: {e}")
    @registry.register(
        name="batch_inference",
        priority=3,
        required=False,  # optional check
        depends_on=["model_loads_correctly"],
    )
    class BatchInferenceCheck(BaseCheck):
        """Run batch inference with 3 samples. Optional check."""
        def execute(self, context):
            runner = context.runner
            if runner is None:
                return CheckResult(passed=False, message="No runner in context")
            try:
                runner.load(context.model_path)
                samples = [
                    IrisInput(features=[5.1, 3.5, 1.4, 0.2]),
                    IrisInput(features=[6.7, 3.1, 4.7, 1.5]),
                    IrisInput(features=[5.8, 2.7, 5.1, 1.9]),
                ]
                results = runner.batch(samples)
                return CheckResult(
                    passed=len(results) == 3,
                    message=f"Batch inference: {len(results)} predictions",
                    details={"predictions": [r.prediction for r in results]},
                )
            except Exception as e:
                return CheckResult(passed=False, message=f"Batch failed: {e}")
    @registry.register(name="metric_threshold_check", priority=4, required=True)
    class MetricThresholdCheck(BaseCheck):
        """
        Verify that current metrics meet the strategy's thresholds.
        Tests: ThresholdStrategy.evaluate() via context.metric_strategy.
        """
        def execute(self, context):
            strategy = context.metric_strategy
            if strategy is None:
                return CheckResult(
                    passed=False, message="No metric_strategy in context"
                )
            # Pull current metrics from extras (set by run_e2e.py)
            current_metrics = context.get("current_eval_metrics", {})
            if not current_metrics:
                return CheckResult(
                    passed=False, message="No current_eval_metrics in context"
                )
            ctx = MetricContext(current=current_metrics, previous=None)
            return strategy.evaluate(ctx)

# ═════════════════════════════════════════════════════════════
# EDGE CASE: Dependency Skip (Phase 5B)
# ═════════════════════════════════════════════════════════════
def register_dependency_skip_checks(registry):
    """Register checks to test dependency-skip behavior."""
    @registry.register(name="always_fails", priority=0, required=True)
    class AlwaysFailsCheck(BaseCheck):
        """Intentionally fails so dependent check is skipped."""
        def execute(self, context):
            return CheckResult(passed=False, message="intentional failure")

    @registry.register(
        name="should_skip",
        priority=1,
        required=True,
        depends_on=["always_fails"],
    )
    class ShouldSkipCheck(BaseCheck):
        """Should be SKIPPED because always_fails didn't pass."""
        def execute(self, context):
            return CheckResult(passed=True, message="should never run")

    @registry.register(name="independent_pass", priority=2, required=False)
    class IndependentPassCheck(BaseCheck):
        """No dependencies — should run and pass even after failures."""
        def execute(self, context):
            return CheckResult(passed=True, message="independent check passed")

# ═════════════════════════════════════════════════════════════
# EDGE CASE: Lifecycle Hooks (Phase 5C)
# ═════════════════════════════════════════════════════════════
def register_lifecycle_checks(registry):
    """Register checks to test before/after/finalize lifecycle hooks."""

    @registry.register(name="lifecycle_hooks", priority=0, required=True)
    class LifecycleHooksCheck(BaseCheck):
        """Tests that before(), after(), and finalize() all fire."""
        def before(self, context):
            context["lifecycle_before_ran"] = True

        def execute(self, context):
            before_ran = context.get("lifecycle_before_ran", False)
            return CheckResult(
                passed=before_ran,
                message=f"before() ran: {before_ran}",
            )

        def after(self, context, result):
            context["lifecycle_after_ran"] = True

        def finalize(self):
            pass  # just verify it doesn't crash

    @registry.register(name="before_raises", priority=1, required=True)
    class BeforeRaisesCheck(BaseCheck):
        """Tests that an exception in before() is caught and reported."""
        def before(self, context):
            raise RuntimeError("intentional before() error")

        def execute(self, context):
            return CheckResult(passed=True, message="should never run")

# ═════════════════════════════════════════════════════════════
# EDGE CASE: Real Metric-From-DB Check (Phase 5D)
# ═════════════════════════════════════════════════════════════
def register_metric_db_checks(registry):
    """Register check that reads metrics from PostgreSQL via context.metric_manager."""

    @registry.register(name="metric_from_db", priority=0, required=True)
    class MetricFromDBCheck(BaseCheck):
        """
        Queries PostgreSQL via context.metric_manager.get_latest_metrics()
        to verify metrics logged in Phase 3 are readable inside a CI check.
        """
        def execute(self, context):
            mm = context.metric_manager
            if mm is None:
                return CheckResult(passed=False, message="No metric_manager in context")
            try:
                latest = mm.get_latest_metrics(run_id=context.run_id)
                if not latest:
                    return CheckResult(
                        passed=False,
                        message="get_latest_metrics returned empty",
                    )
                has_accuracy = "train_accuracy" in latest or "final_accuracy" in latest
                return CheckResult(
                    passed=has_accuracy,
                    message=f"DB metrics retrieved: {sorted(latest.keys())}",
                    details=latest,
                )
            except Exception as e:
                return CheckResult(passed=False, message=f"DB query failed: {e}")

# ═════════════════════════════════════════════════════════════
# EDGE CASE: Baseline Comparison (Phase 5E)
# ═════════════════════════════════════════════════════════════
def register_baseline_checks(registry, current_metrics, previous_metrics=None):
    """Register check using BaselineComparisonStrategy."""

    @registry.register(name="baseline_comparison", priority=0, required=True)
    class BaselineComparisonCheck(BaseCheck):
        """Tests BaselineComparisonStrategy with optional previous metrics."""
        def execute(self, context):
            strategy = BaselineComparisonStrategy(
                metrics_to_compare=["accuracy"],
                min_delta=0.0,
                fallback_thresholds={"accuracy": 0.50},
            )
            ctx = MetricContext(
                current=current_metrics,
                previous=previous_metrics,
            )
            return strategy.evaluate(ctx)

# ═════════════════════════════════════════════════════════════
# EDGE CASE: Optional Failure (Phase 5G)
# ═════════════════════════════════════════════════════════════
def register_optional_failure_checks(registry):
    """Register checks to verify optional failures don't block CI."""

    @registry.register(name="required_pass", priority=0, required=True)
    class RequiredPassCheck(BaseCheck):
        """A passing required check."""
        def execute(self, context):
            return CheckResult(passed=True, message="required check passed")

    @registry.register(name="optional_failure", priority=1, required=False)
    class OptionalFailureCheck(BaseCheck):
        """An optional check that fails — should NOT block CI."""
        def execute(self, context):
            return CheckResult(passed=False, message="optional check intentionally failed")

# ═════════════════════════════════════════════════════════════
# EDGE CASE: Threshold Missing Metric (Phase 5H)
# ═════════════════════════════════════════════════════════════
def register_threshold_missing_checks(registry):
    """Register check to verify ThresholdStrategy behavior with missing metrics."""
    from src.mlforge.core.deployment_manager.ci.metrics import ThresholdStrategy

    @registry.register(name="threshold_missing_metric", priority=0, required=True)
    class ThresholdMissingMetricCheck(BaseCheck):
        """ThresholdStrategy should fail when a required metric key is missing."""
        def execute(self, context):
            strategy = ThresholdStrategy({"nonexistent_metric": 0.90})
            current = context.get("current_eval_metrics", {"accuracy": 0.95})
            ctx = MetricContext(current=current, previous=None)
            result = strategy.evaluate(ctx)
            # We EXPECT this to fail — strategy should report missing metric
            if not result.passed and "not found" in result.message:
                return CheckResult(
                    passed=True,
                    message=f"Correctly detected missing metric: {result.message}",
                )
            return CheckResult(
                passed=False,
                message=f"Expected failure for missing metric, got: passed={result.passed}, msg={result.message}",
            )
