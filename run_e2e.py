"""
MLForge E2E Test — User Orchestration Script
=============================================
This is what a real user would write to use MLForge.
Tests all managers with real backends:
  - env_manager    → LocalMachineEnvironment  (local)
  - metric_manager → PostgreSQLTracker        (PostgreSQL)
  - model_manager  → MongoRegistry            (MongoDB)
  - CI runner      → LocalCI + PipelineManager (multiple edge cases)
Usage:
    set MLFORGE_ROOT=e:\\MLF\\MLForge
    set MLFORGE_PG_PASSWORD=your_password
    python run_e2e.py
"""
import os
import sys
import shutil
import socket
import json
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime
# ─── Point to MLForge source ───────────────────────────────
# A real user would `pip install mlforge` instead of these two lines.
MLFORGE_ROOT = os.getenv("MLFORGE_ROOT", r"e:\MLF\MLForge")
sys.path.insert(0, MLFORGE_ROOT)
sys.path.insert(0, os.path.join(MLFORGE_ROOT, "src"))
# ─── Third-party imports ───────────────────────────────────
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# ─── MLForge imports ───────────────────────────────────────
from src.mlforge.core.env_manager import LocalMachineEnvironment
from mlforge.core.metric_manager import PostgreSQLTracker
from mlforge.core.model_manager.mongo import MongoRegistry
from mlforge.core.schema import ArtifactContext, ArtifactContextCreate
from mlforge.core.schemas import ModelCheckpointSchema
from mlforge.utils.db import ArtifactContextDB
# CI imports
from src.mlforge.core.deployment_manager.ci.local_ci import LocalCI
from src.mlforge.core.deployment_manager.ci.registry import CheckRegistry
from src.mlforge.core.deployment_manager.ci.context import CIContext
from src.mlforge.core.deployment_manager.ci.inference import InferenceLogger
from src.mlforge.core.deployment_manager.ci.metrics import ThresholdStrategy
from src.mlforge.core.deployment_manager.ci.pipeline import PipelineManager
from src.mlforge.core.deployment_manager.ci.report_writer import save_report
# User-defined checks & inference runner (from this repo)
from ci_checks import (
    IrisInferenceRunner,
    register_all_checks,
    register_dependency_skip_checks,
    register_lifecycle_checks,
    register_metric_db_checks,
    register_baseline_checks,
    register_optional_failure_checks,
    register_threshold_missing_checks,
)
# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════
# The remote repo URL — this repo itself (change to your URL after pushing)
GIT_URL    = os.getenv("MLFORGE_GIT_URL", "https://github.com/sidd-boop/mlforge-demo.git")
GIT_BRANCH = os.getenv("MLFORGE_GIT_BRANCH", "main")
# PostgreSQL
PG_HOST     = os.getenv("MLFORGE_PG_HOST", "localhost")
PG_PORT     = int(os.getenv("MLFORGE_PG_PORT", "5432"))
PG_DATABASE = os.getenv("MLFORGE_PG_DATABASE", "mlforge_test")
PG_USER     = os.getenv("MLFORGE_PG_USER", "postgres")
PG_PASSWORD = os.getenv("MLFORGE_PG_PASSWORD", "12345678")
# MongoDB
MONGO_URI        = os.getenv("MLFORGE_MONGO_URI", "mongodb://localhost:27017")
MONGO_DATABASE   = os.getenv("MLFORGE_MONGO_DATABASE", "mlforge_e2e")
MONGO_COLLECTION = f"models_e2e_{uuid4().hex[:8]}"   # unique per run to avoid conflicts
# Workspace — fresh temp directory for env_manager
WORKSPACE = Path(tempfile.mkdtemp(prefix="mlforge_e2e_"))
# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def is_port_open(host, port, timeout=2.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0
def cleanup(path):
    def _handle(func, p, exc_info):
        os.chmod(p, 0o777)
        func(p)
    if path.exists():
        shutil.rmtree(path, onerror=_handle)
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

passed_phases = []
def phase_pass(name, details=""):
    passed_phases.append(name)
    print(f"\n✓ {name}: ALL PASSED")
    if details:
        print(details)

# ════════════════════════════════════════════════════════════
# SHARED CONTEXT (one run_id/experiment_id used everywhere)
# ════════════════════════════════════════════════════════════
experiment_id = uuid4()
run_id        = uuid4()
user_id       = uuid4()
shared_ctx = ArtifactContext(
    run_id=run_id,
    experiment_id=experiment_id,
    user_id=user_id,
)
print(f"Shared context:")
print(f"  run_id:        {run_id}")
print(f"  experiment_id: {experiment_id}")
print(f"  user_id:       {user_id}")
print(f"  workspace:     {WORKSPACE}")
# ════════════════════════════════════════════════════════════
# PHASE 1 : ENVIRONMENT MANAGER (LocalMachineEnvironment)
# ════════════════════════════════════════════════════════════
section("PHASE 1: Environment Manager (Local)")
local_env = LocalMachineEnvironment(
    env_path=".env",
    config={
        "project_name": "mlforge-e2e-test",
        "git_url":       GIT_URL,
        "branch":        GIT_BRANCH,
        "root_dir":      str(WORKSPACE),
        "working_dir":   "working",
        "input_dir":     "input",
        "output_dir":    "output",
        "device":        "cpu",
        "skip_install":  False,        # ← tests install_packages()
    },
)
# initialize() calls: connect → setup_directory_structure → sync_code
#                      → load_environment_variables → install_packages → setup_device
local_env.initialize()
# ── Assertions ──────────────────────────────────────────────
assert local_env.is_ready,                        "env_manager: setup failed"
assert local_env.device == "cpu",                 "env_manager: device should be cpu"
assert local_env.working_dir.exists(),            "env_manager: working_dir missing"
assert local_env.input_dir.exists(),              "env_manager: input_dir missing"
assert local_env.output_dir.exists(),             "env_manager: output_dir missing"
assert (local_env.working_dir / "train.py").exists(), "env_manager: train.py not cloned"
# .env should have been loaded and securely deleted
assert os.environ.get("MLFORGE_E2E_TEST") == "enabled", "env_manager: .env not loaded"
assert not (local_env.working_dir / ".env").exists(),    "env_manager: .env not deleted"
phase_pass("env_manager",
    "  Features: connect, setup_directory_structure, sync_code,\n"
    "            load_environment_variables, install_packages, setup_device, validate_setup")
# ════════════════════════════════════════════════════════════
# PHASE 2 : TRAIN A MODEL
# ════════════════════════════════════════════════════════════
section("PHASE 2: Training Model")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y,
)
classes = np.unique(y)
clf = SGDClassifier(loss="log_loss", random_state=42)
epochs = 6
accuracy_history = []
for epoch in range(1, epochs + 1):
    clf.partial_fit(X_train, y_train, classes=classes)
    acc = float(accuracy_score(y_test, clf.predict(X_test)))
    accuracy_history.append(acc)
final_preds    = clf.predict(X_test)
final_accuracy = float(accuracy_score(y_test, final_preds))
cm             = confusion_matrix(y_test, final_preds).tolist()
# Save model to workspace/output
model_dir = WORKSPACE / "output" / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "sgd_iris.joblib"
joblib.dump(clf, str(model_path))
print(f"✓ Model trained: {epochs} epochs, accuracy={final_accuracy:.4f}")
print(f"✓ Model saved to {model_path}")
# ════════════════════════════════════════════════════════════
# PHASE 3 : METRIC MANAGER (PostgreSQLTracker)
# ════════════════════════════════════════════════════════════
section("PHASE 3: Metric Manager (PostgreSQL)")
assert PG_PASSWORD, "Set MLFORGE_PG_PASSWORD before running"
assert is_port_open(PG_HOST, PG_PORT), f"PostgreSQL not reachable at {PG_HOST}:{PG_PORT}"
# 3a. Persist artifact context via ArtifactContextDB
artifact_db = ArtifactContextDB({
    "host": PG_HOST, "port": PG_PORT,
    "database": PG_DATABASE, "user": PG_USER, "password": PG_PASSWORD,
})
artifact_db.connect()
try:
    artifact_db.initialize_schema(os.path.join(MLFORGE_ROOT, "database", "schema.sql"))
    persisted = artifact_db.upsert_artifact_context(
        ArtifactContextCreate(
            run_id=shared_ctx.run_id,
            experiment_id=shared_ctx.experiment_id,
            user_id=shared_ctx.user_id,
        )
    )
    assert persisted.run_id == shared_ctx.run_id
finally:
    artifact_db.disconnect()
print("✓ ArtifactContextDB: connect, initialize_schema, upsert — PASSED")
# 3b. PostgreSQLTracker — log all metric types
pg_cfg = {
    "host": PG_HOST, "port": PG_PORT,
    "database": PG_DATABASE, "username": PG_USER, "password": PG_PASSWORD,
    "buffer_size": 5, "schema": "public", "table_name": "metrics",
}
pg_tracker = PostgreSQLTracker(pg_cfg, shared_ctx)
pg_tracker.connect()
try:
    # ── log_metric (scalar, per epoch) ──
    for epoch, acc in enumerate(accuracy_history, 1):
        pg_tracker.log_metric("train_accuracy", acc, step=epoch, category="training")
    # ── log_metrics (multiple scalars at once) ──
    pg_tracker.log_metrics(
        {"final_accuracy": final_accuracy, "num_classes": float(len(classes))},
        step=epochs,
        category="evaluation",
    )
    # ── log_image (confusion matrix) ──
    pg_tracker.log_image("confusion_matrix", cm, step=epochs, category="evaluation")
    # ── log_histogram (prediction distribution) ──
    pg_tracker.log_histogram("prediction_distribution", final_preds.tolist(), step=epochs)
    # ── log_custom_event (arbitrary metadata) ──
    pg_tracker.log_custom_event(
        "run_summary",
        {"framework": "mlforge", "epochs": epochs, "model": "SGDClassifier"},
        metric_type="summary",
        step=epochs,
        category="system",
    )
    # ── phase() context manager ──
    with pg_tracker.phase("validation"):
        pg_tracker.log_metric("val_accuracy", final_accuracy, step=epochs)
    # Force flush remaining buffer
    pg_tracker._flush_buffer()
    # ── get_metrics — various filters ──
    all_metrics       = pg_tracker.get_metrics(run_id=str(run_id))
    train_acc_metrics = pg_tracker.get_metrics(run_id=str(run_id), metric_key="train_accuracy")
    eval_metrics      = pg_tracker.get_metrics(run_id=str(run_id), category="evaluation")
    step_range        = pg_tracker.get_metrics(run_id=str(run_id), metric_key="train_accuracy", step_range=(1, 3))
    by_type           = pg_tracker.get_metrics(run_id=str(run_id), metric_type="scalar")
    # ── get_latest_metrics ──
    latest = pg_tracker.get_latest_metrics(run_id=str(run_id))
    # ── compare_runs ──
    comparison = pg_tracker.compare_runs("train_accuracy", [str(run_id)])
finally:
    pg_tracker.disconnect()
# ── Assertions ──
assert len(train_acc_metrics) == epochs,          f"Expected {epochs} train_accuracy metrics, got {len(train_acc_metrics)}"
assert "final_accuracy" in latest,                "final_accuracy not in latest"
assert "confusion_matrix" in latest,              "confusion_matrix not in latest"
assert "run_summary" in latest,                   "run_summary not in latest"
assert "val_accuracy" in latest,                  "val_accuracy (from phase) not in latest"
assert "prediction_distribution" in latest,       "histogram not in latest"
assert len(step_range) == 3,                      f"Step range 1-3 should return 3, got {len(step_range)}"
assert str(run_id) in comparison,                 "compare_runs missing current run"
assert len(eval_metrics) >= 2,                    "Should have at least 2 eval metrics"
phase_pass("PostgreSQLTracker",
    f"  Total metrics logged:   {len(all_metrics)}\n"
    f"  Latest keys:            {sorted(latest.keys())}\n"
    f"  Comparison runs:        {list(comparison.keys())}\n"
    f"  Features tested:        log_metric, log_metrics, log_image, log_histogram,\n"
    f"                          log_custom_event, phase(), get_metrics (5 filter combos),\n"
    f"                          get_latest_metrics, compare_runs")
# ════════════════════════════════════════════════════════════
# PHASE 4 : MODEL MANAGER (MongoRegistry)
# ════════════════════════════════════════════════════════════
section("PHASE 4: Model Manager (MongoDB)")
assert is_port_open("localhost", 27017), "MongoDB not reachable at localhost:27017"
mongo_cfg = {
    "connection_string": MONGO_URI,
    "database": MONGO_DATABASE,
    "collection": MONGO_COLLECTION,
    "experiment_id": shared_ctx.experiment_id,
    "run_id": shared_ctx.run_id,
    "user_id": shared_ctx.user_id,
}
model_id   = uuid4()
version_id = f"sgd-iris-{run_id.hex[:8]}"
checkpoint = ModelCheckpointSchema(
    experiment_id=shared_ctx.experiment_id,
    run_id=shared_ctx.run_id,
    user_id=shared_ctx.user_id,
    start_timestamp=datetime.now(),
    version_id=version_id,
    artifact_uri=str(model_path),
    model_id=model_id,
    epoch=epochs,
    global_step=epochs * len(X_train),
    model_arch="SGDClassifier",
    optimizer_info={"loss": "log_loss", "epochs": epochs, "lr": "default"},
)
mongo = MongoRegistry(mongo_cfg)
mongo.initialize()
try:
    # ── set_artifact_context ──
    mongo.set_artifact_context(shared_ctx.experiment_id, shared_ctx.run_id, shared_ctx.user_id)
    # ── log_model ──
    logged_version = mongo.log_model(checkpoint)
    assert logged_version == version_id, f"Expected {version_id}, got {logged_version}"
    # ── get_model ──
    retrieved = mongo.get_model(version_id)
    assert retrieved is not None,                          "get_model returned None"
    assert retrieved.version_id == version_id,             "version_id mismatch"
    assert retrieved.run_id == shared_ctx.run_id,          "run_id mismatch"
    assert retrieved.artifact_uri == str(model_path),      "artifact_uri mismatch"
    assert retrieved.epoch == epochs,                      "epoch mismatch"
    assert retrieved.model_arch == "SGDClassifier",        "model_arch mismatch"
    # ── view_models — query by run_id ──
    by_run = mongo.view_models({"run_id": shared_ctx.run_id})
    assert len(by_run) >= 1, "view_models by run_id failed"
    # ── view_models — query by experiment_id ──
    by_exp = mongo.view_models({"experiment_id": shared_ctx.experiment_id})
    assert len(by_exp) >= 1, "view_models by experiment_id failed"
    # ── view_models — all ──
    all_models = mongo.view_models()
    assert len(all_models) >= 1, "view_models() returned nothing"
    # ── log a second model checkpoint ──
    version_id_2 = f"sgd-iris-v2-{uuid4().hex[:8]}"
    checkpoint_2 = ModelCheckpointSchema(
        experiment_id=shared_ctx.experiment_id,
        run_id=shared_ctx.run_id,
        user_id=shared_ctx.user_id,
        start_timestamp=datetime.now(),
        version_id=version_id_2,
        artifact_uri=str(model_path),
        model_id=uuid4(),
        epoch=epochs + 2,
        global_step=(epochs + 2) * len(X_train),
        model_arch="SGDClassifier",
        optimizer_info={"loss": "log_loss", "epochs": epochs + 2},
    )
    mongo.log_model(checkpoint_2)
    multi = mongo.view_models({"experiment_id": shared_ctx.experiment_id})
    assert len(multi) >= 2, f"Expected ≥2 models, got {len(multi)}"
finally:
    mongo.disconnect()
phase_pass("MongoRegistry",
    f"  Models logged:   2 ({version_id}, {version_id_2})\n"
    f"  Features tested: initialize, connect, set_artifact_context, log_model,\n"
    f"                   get_model, view_models (by run, by experiment, all), disconnect")

# ════════════════════════════════════════════════════════════
# PHASE 5A : CI — HAPPY PATH (all checks pass)
# ════════════════════════════════════════════════════════════
section("PHASE 5A: CI Happy Path")
registry_5a = CheckRegistry()
register_all_checks(registry_5a)

inference_logger_5a = InferenceLogger()
runner_5a = IrisInferenceRunner(inference_logger_5a)

ci_context_5a = CIContext(
    model_path=str(model_path),
    model_name="sgd-iris-e2e",
    runner=runner_5a,
    inference_logger=inference_logger_5a,
    inference_sample={"features": [5.1, 3.5, 1.4, 0.2]},
    version_id=version_id,
    run_id=str(run_id),
    metric_strategy=ThresholdStrategy({"accuracy": 0.50}),
    extras={
        "current_eval_metrics": {
            "accuracy": final_accuracy,
            "num_classes": float(len(classes)),
        }
    },
)

ci_config_5a = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci_5a = LocalCI(config=ci_config_5a, registry=registry_5a)
pipeline_5a = PipelineManager(ci=ci_5a, context=ci_context_5a)
result_5a = pipeline_5a.run()

report_5a = result_5a.ci_result
print(report_5a.summary())
report_path = str(WORKSPACE / "output" / "ci_report.json")
save_report(report_5a, report_path)

# ── Assertions ─────────────────────────────────────────────
assert len(report_5a.checks) >= 5,             f"Expected ≥5 checks, got {len(report_5a.checks)}"
assert len(report_5a.passed_checks) >= 4,      f"Expected ≥4 passed, got {len(report_5a.passed_checks)}"
assert len(report_5a.pipeline_logs) > 0,       "No pipeline logs captured"
assert len(report_5a.inference_logs) > 0,      "No inference logs captured"
assert os.path.exists(report_path),            "CI report JSON not saved"
exit_code_5a = pipeline_5a.get_exit_code()
assert exit_code_5a == 0,                      f"Expected exit code 0, got {exit_code_5a}"
assert result_5a.passed,                       f"CI pipeline failed!\n{report_5a.summary()}"
# Verify report JSON
with open(report_path) as f:
    report_data = json.load(f)
assert report_data["passed"] is True
assert len(report_data["checks"]) >= 5
phase_pass("CI Happy Path",
    f"  Checks: {len(report_5a.checks)} run, {len(report_5a.passed_checks)} passed, "
    f"{len(report_5a.failed_checks)} failed, {len(report_5a.skipped_checks)} skipped\n"
    f"  Exit code: {exit_code_5a}\n"
    f"  Features: CheckRegistry, @register, BaseCheck, CheckRunner,\n"
    f"            LocalCI, PipelineManager, InferenceLogger, ThresholdStrategy, save_report")

# ════════════════════════════════════════════════════════════
# PHASE 5B : CI — DEPENDENCY SKIP + FAIL BEHAVIOR
# ════════════════════════════════════════════════════════════
section("PHASE 5B: CI Dependency Skip + Fail Behavior")
registry_5b = CheckRegistry()
register_dependency_skip_checks(registry_5b)

ci_config_5b = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,       # ← run ALL checks even after failure
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci_5b = LocalCI(config=ci_config_5b, registry=registry_5b)
pipeline_5b = PipelineManager(
    ci=ci_5b,
    context=CIContext(model_path=str(model_path), model_name="dep-skip-test"),
)
result_5b = pipeline_5b.run()

report_5b = result_5b.ci_result
print(report_5b.summary())

# ── Assertions ─────────────────────────────────────────────
# Pipeline should FAIL because always_fails is required
assert not result_5b.passed,               "Expected pipeline to fail (always_fails is required)"
assert pipeline_5b.get_exit_code() == 1,   "Expected exit code 1"

# should_skip must be skipped, not failed
skipped_names = [c.name for c in report_5b.skipped_checks]
assert "should_skip" in skipped_names,     f"should_skip not in skipped: {skipped_names}"

# always_fails must be in failed_checks (not skipped)
failed_names = [c.name for c in report_5b.failed_checks]
assert "always_fails" in failed_names,     f"always_fails not in failed: {failed_names}"

# independent_pass should still run and pass (no dependency, fail_fast=False)
passed_names = [c.name for c in report_5b.passed_checks]
assert "independent_pass" in passed_names, f"independent_pass not in passed: {passed_names}"

# Verify total check count
assert len(report_5b.checks) == 3,         f"Expected 3 checks, got {len(report_5b.checks)}"

phase_pass("CI Dependency Skip",
    f"  always_fails:     FAILED (as expected)\n"
    f"  should_skip:      SKIPPED (dependency not met)\n"
    f"  independent_pass: PASSED (no dependency, ran despite failures)\n"
    f"  Pipeline passed:  {result_5b.passed} (correct: False)\n"
    f"  Exit code:        {pipeline_5b.get_exit_code()}\n"
    f"  Features tested:  depends_on, SkippedResult, fail_fast=False, dependency resolution")

# ════════════════════════════════════════════════════════════
# PHASE 5C : CI — LIFECYCLE HOOKS (before/after/finalize)
# ════════════════════════════════════════════════════════════
section("PHASE 5C: CI Lifecycle Hooks")
registry_5c = CheckRegistry()
register_lifecycle_checks(registry_5c)

ci_config_5c = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci_5c = LocalCI(config=ci_config_5c, registry=registry_5c)
ctx_5c = CIContext(model_path=str(model_path), model_name="lifecycle-test")
pipeline_5c = PipelineManager(ci=ci_5c, context=ctx_5c)
result_5c = pipeline_5c.run()

report_5c = result_5c.ci_result
print(report_5c.summary())

# ── Assertions ─────────────────────────────────────────────
assert len(report_5c.checks) == 2,         f"Expected 2 checks, got {len(report_5c.checks)}"

# lifecycle_hooks should pass — before() set data, execute() read it
lifecycle_check = next(c for c in report_5c.checks if c.name == "lifecycle_hooks")
assert lifecycle_check.passed,             f"lifecycle_hooks failed: {lifecycle_check.message}"

# before_raises should FAIL with "before() raised" in its message
before_raises_check = next(c for c in report_5c.checks if c.name == "before_raises")
assert not before_raises_check.passed,     "before_raises should have failed"
assert "before() raised" in before_raises_check.message, \
    f"Expected 'before() raised' in message, got: {before_raises_check.message}"

phase_pass("CI Lifecycle Hooks",
    f"  lifecycle_hooks:  PASSED (before→execute→after→finalize all fired)\n"
    f"  before_raises:    FAILED (exception caught, reported correctly)\n"
    f"  Features tested:  BaseCheck.before(), after(), finalize(), _run() error handling")

# ════════════════════════════════════════════════════════════
# PHASE 5D : CI — REAL METRIC-FROM-DB CHECK
# ════════════════════════════════════════════════════════════
section("PHASE 5D: CI Metric-From-DB Check")
registry_5d = CheckRegistry()
register_metric_db_checks(registry_5d)

# Reconnect pg_tracker for the CI check to use
pg_tracker_5d = PostgreSQLTracker(pg_cfg, shared_ctx)
pg_tracker_5d.connect()

ci_config_5d = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci_5d = LocalCI(config=ci_config_5d, registry=registry_5d)
ctx_5d = CIContext(
    model_path=str(model_path),
    model_name="metric-db-test",
    metric_manager=pg_tracker_5d,
    run_id=str(run_id),
)
pipeline_5d = PipelineManager(ci=ci_5d, context=ctx_5d)
result_5d = pipeline_5d.run()

report_5d = result_5d.ci_result
print(report_5d.summary())
pg_tracker_5d.disconnect()

# ── Assertions ─────────────────────────────────────────────
db_check = next(c for c in report_5d.checks if c.name == "metric_from_db")
assert db_check.passed,                    f"metric_from_db failed: {db_check.message}"
assert result_5d.passed,                   "Pipeline should pass"

phase_pass("CI Metric-From-DB",
    f"  metric_from_db: PASSED — real PostgreSQL read inside a CI check\n"
    f"  Message:        {db_check.message}\n"
    f"  Features tested: context.metric_manager, get_latest_metrics() inside BaseCheck")

# ════════════════════════════════════════════════════════════
# PHASE 5E : CI — BASELINE COMPARISON STRATEGY
# ════════════════════════════════════════════════════════════
section("PHASE 5E: CI BaselineComparisonStrategy")

# Test 1: No previous metrics → fallback to thresholds
print("  Test 1: No previous → fallback thresholds")
registry_5e1 = CheckRegistry()
register_baseline_checks(
    registry_5e1,
    current_metrics={"accuracy": final_accuracy},
    previous_metrics=None,    # ← no previous = first-time deploy
)
ci_5e1 = LocalCI(
    config={"model_name": "baseline-test", "model_path": str(model_path),
            "fail_fast": False, "check_timeout": 30, "pipeline_timeout": 120},
    registry=registry_5e1,
)
pipeline_5e1 = PipelineManager(
    ci=ci_5e1,
    context=CIContext(model_path=str(model_path), model_name="baseline-no-prev"),
)
result_5e1 = pipeline_5e1.run()
report_5e1 = result_5e1.ci_result
print(report_5e1.summary())
baseline_check_1 = next(c for c in report_5e1.checks if c.name == "baseline_comparison")
assert baseline_check_1.passed,            f"Baseline (no previous) should pass: {baseline_check_1.message}"

# Test 2: With previous metrics — new model same as old (delta=0.0 required, should pass)
print("\n  Test 2: With previous → comparison")
registry_5e2 = CheckRegistry()
register_baseline_checks(
    registry_5e2,
    current_metrics={"accuracy": final_accuracy},
    previous_metrics={"accuracy": final_accuracy - 0.01},   # ← current beats previous
)
ci_5e2 = LocalCI(
    config={"model_name": "baseline-test", "model_path": str(model_path),
            "fail_fast": False, "check_timeout": 30, "pipeline_timeout": 120},
    registry=registry_5e2,
)
pipeline_5e2 = PipelineManager(
    ci=ci_5e2,
    context=CIContext(model_path=str(model_path), model_name="baseline-with-prev"),
)
result_5e2 = pipeline_5e2.run()
report_5e2 = result_5e2.ci_result
print(report_5e2.summary())
baseline_check_2 = next(c for c in report_5e2.checks if c.name == "baseline_comparison")
assert baseline_check_2.passed,            f"Baseline (with previous) should pass: {baseline_check_2.message}"

phase_pass("CI BaselineComparison",
    f"  No previous:   PASSED (fallback thresholds used)\n"
    f"  With previous: PASSED (current beats baseline)\n"
    f"  Features tested: BaselineComparisonStrategy, fallback_thresholds, min_delta, MetricContext")

# ════════════════════════════════════════════════════════════
# PHASE 5F : CI — DRY RUN MODE
# ════════════════════════════════════════════════════════════
section("PHASE 5F: CI Dry Run Mode")
registry_5f = CheckRegistry()
register_all_checks(registry_5f)  # register checks — but dry_run should skip them all

ci_config_5f = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,
    "check_timeout":     30,
    "pipeline_timeout":  120,
    "dry_run":           True,           # ← SKIP all checks
}
ci_5f = LocalCI(config=ci_config_5f, registry=registry_5f)
pipeline_5f = PipelineManager(
    ci=ci_5f,
    context=CIContext(model_path=str(model_path), model_name="dry-run-test"),
)
result_5f = pipeline_5f.run()

report_5f = result_5f.ci_result

# ── Assertions ─────────────────────────────────────────────
assert result_5f.passed,                   "Dry run should always pass"
assert len(report_5f.checks) == 0,         f"Dry run should have 0 checks, got {len(report_5f.checks)}"
assert pipeline_5f.get_exit_code() == 0,   "Dry run exit code should be 0"

phase_pass("CI Dry Run",
    f"  Pipeline passed: True (no checks executed)\n"
    f"  Checks run:      0\n"
    f"  Exit code:       0\n"
    f"  Features tested: dry_run=True bypasses all checks, report still built")

# ════════════════════════════════════════════════════════════
# PHASE 5G : CI — OPTIONAL FAILURE DOESN'T BLOCK
# ════════════════════════════════════════════════════════════
section("PHASE 5G: CI Optional Failure")
registry_5g = CheckRegistry()
register_optional_failure_checks(registry_5g)

ci_config_5g = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci_5g = LocalCI(config=ci_config_5g, registry=registry_5g)
pipeline_5g = PipelineManager(
    ci=ci_5g,
    context=CIContext(model_path=str(model_path), model_name="optional-fail-test"),
)
result_5g = pipeline_5g.run()

report_5g = result_5g.ci_result
print(report_5g.summary())

# ── Assertions ─────────────────────────────────────────────
assert result_5g.passed,                   "Pipeline should PASS (only optional check failed)"
assert pipeline_5g.get_exit_code() == 0,   "Exit code should be 0"

# required_pass should be in passed
passed_5g = [c.name for c in report_5g.passed_checks]
assert "required_pass" in passed_5g,       f"required_pass not in passed: {passed_5g}"

# optional_failure should be in failed but NOT block CI
failed_5g = [c.name for c in report_5g.failed_checks]
assert "optional_failure" in failed_5g,    f"optional_failure not in failed: {failed_5g}"

phase_pass("CI Optional Failure",
    f"  required_pass:    PASSED\n"
    f"  optional_failure: FAILED (but did not block CI)\n"
    f"  Pipeline passed:  True\n"
    f"  Exit code:        0\n"
    f"  Features tested:  required=False, optional failure isolation")

# ════════════════════════════════════════════════════════════
# PHASE 5H : CI — THRESHOLD MISSING METRIC
# ════════════════════════════════════════════════════════════
section("PHASE 5H: CI Threshold Missing Metric")
registry_5h = CheckRegistry()
register_threshold_missing_checks(registry_5h)

ci_config_5h = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci_5h = LocalCI(config=ci_config_5h, registry=registry_5h)
pipeline_5h = PipelineManager(
    ci=ci_5h,
    context=CIContext(
        model_path=str(model_path),
        model_name="threshold-missing-test",
        extras={"current_eval_metrics": {"accuracy": final_accuracy}},
    ),
)
result_5h = pipeline_5h.run()

report_5h = result_5h.ci_result
print(report_5h.summary())

# ── Assertions ─────────────────────────────────────────────
threshold_check = next(c for c in report_5h.checks if c.name == "threshold_missing_metric")
assert threshold_check.passed, \
    f"threshold_missing_metric should PASS (correctly detected missing): {threshold_check.message}"

phase_pass("CI Threshold Missing Metric",
    f"  ThresholdStrategy correctly reported missing metric key\n"
    f"  Check result: PASSED (meta-check verifies error detection)\n"
    f"  Features tested: ThresholdStrategy with nonexistent key, graceful failure message")

# ════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════
section("CLEANUP")
local_env.teardown()
cleanup(WORKSPACE)
print(f"✓ Workspace cleaned: {WORKSPACE}")

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  🎉 ALL E2E TESTS PASSED 🎉")
print(f"{'='*60}")
print(f"\n  Phases passed: {len(passed_phases)}")
for p in passed_phases:
    print(f"    ✓ {p}")
print(f"\n  CI Edge Cases Covered:")
print(f"    • Happy path (5 checks pass)")
print(f"    • Dependency skip (SkippedResult)")
print(f"    • Lifecycle hooks (before/after/finalize)")
print(f"    • Real DB metric read inside check")
print(f"    • BaselineComparisonStrategy (no prev + with prev)")
print(f"    • Dry run mode (skip all checks)")
print(f"    • Optional failure doesn't block CI")
print(f"    • Missing metric key in ThresholdStrategy")
print(f"{'='*60}")
