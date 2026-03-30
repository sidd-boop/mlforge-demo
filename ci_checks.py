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
# ─── CI Checks ──────────────────────────────────────────────
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
