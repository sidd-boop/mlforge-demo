"""
MLForge E2E Test — User Orchestration Script
=============================================
This is what a real user would write to use MLForge.
Tests all managers with real backends:
  - env_manager    → LocalMachineEnvironment  (local)
  - metric_manager → PostgreSQLTracker        (PostgreSQL)
  - model_manager  → MongoRegistry            (MongoDB)
  - CI runner      → LocalCI + PipelineManager
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
from ci_checks import IrisInferenceRunner, register_all_checks
# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════
# The remote repo URL — this repo itself (change to your URL after pushing)
GIT_URL    = os.getenv("MLFORGE_GIT_URL", "https://github.com/<YOUR_USERNAME>/mlforge-e2e-test.git")
GIT_BRANCH = os.getenv("MLFORGE_GIT_BRANCH", "main")
# PostgreSQL
PG_HOST     = os.getenv("MLFORGE_PG_HOST", "localhost")
PG_PORT     = int(os.getenv("MLFORGE_PG_PORT", "5432"))
PG_DATABASE = os.getenv("MLFORGE_PG_DATABASE", "mlforge_test")
PG_USER     = os.getenv("MLFORGE_PG_USER", "postgres")
PG_PASSWORD = os.getenv("MLFORGE_PG_PASSWORD", "")
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
        shutil.rmtree(path, onexc=_handle)
def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
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
print("\n✓ env_manager: connect, setup_directory_structure, sync_code,")
print("  load_environment_variables, install_packages, setup_device, validate_setup — ALL PASSED")
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
print(f"\n✓ PostgreSQLTracker: ALL PASSED")
print(f"  Total metrics logged:   {len(all_metrics)}")
print(f"  Latest keys:            {sorted(latest.keys())}")
print(f"  Comparison runs:        {list(comparison.keys())}")
print(f"  Features tested:        log_metric, log_metrics, log_image, log_histogram,")
print(f"                          log_custom_event, phase(), get_metrics (5 filter combos),")
print(f"                          get_latest_metrics, compare_runs")
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
print(f"\n✓ MongoRegistry: ALL PASSED")
print(f"  Models logged:   2 ({version_id}, {version_id_2})")
print(f"  Features tested: initialize, connect, set_artifact_context, log_model,")
print(f"                   get_model, view_models (by run, by experiment, all), disconnect")
# ════════════════════════════════════════════════════════════
# PHASE 5 : CI RUNNER (LocalCI + PipelineManager)
# ════════════════════════════════════════════════════════════
section("PHASE 5: CI Runner (LocalCI + PipelineManager)")
# ── Register checks ────────────────────────────────────────
registry = CheckRegistry()
register_all_checks(registry)
# ── Create inference logger & runner ───────────────────────
inference_logger = InferenceLogger()
runner = IrisInferenceRunner(inference_logger)
# ── Build CI context ───────────────────────────────────────
ci_context = CIContext(
    model_path=str(model_path),
    model_name="sgd-iris-e2e",
    runner=runner,
    inference_logger=inference_logger,
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
# ── Run the pipeline ───────────────────────────────────────
ci_config = {
    "model_name":        "sgd-iris-e2e",
    "model_path":        str(model_path),
    "fail_fast":         False,       # run ALL checks even if one fails
    "check_timeout":     30,
    "pipeline_timeout":  120,
}
ci = LocalCI(config=ci_config, registry=registry)
pipeline = PipelineManager(ci=ci, context=ci_context)
result = pipeline.run()
# ── Reports ────────────────────────────────────────────────
report = result.ci_result
print(report.summary())
report_path = str(WORKSPACE / "output" / "ci_report.json")
save_report(report, report_path)
# ── Assertions ─────────────────────────────────────────────
assert len(report.checks) >= 5,             f"Expected ≥5 checks, got {len(report.checks)}"
assert len(report.passed_checks) >= 4,      f"Expected ≥4 passed, got {len(report.passed_checks)}"
assert len(report.pipeline_logs) > 0,       "No pipeline logs captured"
assert len(report.inference_logs) > 0,      "No inference logs captured"
assert os.path.exists(report_path),         "CI report JSON not saved"
exit_code = pipeline.get_exit_code()
assert exit_code == 0,                      f"Expected exit code 0, got {exit_code}"
assert result.passed,                       f"CI pipeline failed!\n{report.summary()}"
# Verify report JSON
with open(report_path) as f:
    report_data = json.load(f)
assert report_data["passed"] is True
assert len(report_data["checks"]) >= 5
print(f"\n✓ CI Runner: ALL PASSED")
print(f"  Checks registered: {len(registry)}")
print(f"  Checks run:        {len(report.checks)}")
print(f"  Passed:            {len(report.passed_checks)}")
print(f"  Failed:            {len(report.failed_checks)}")
print(f"  Skipped:           {len(report.skipped_checks)}")
print(f"  Pipeline logs:     {len(report.pipeline_logs)}")
print(f"  Inference logs:    {len(report.inference_logs)}")
print(f"  Exit code:         {exit_code}")
print(f"  Report saved:      {report_path}")
print(f"  Features tested:   CheckRegistry, @register (name/priority/required/depends_on),")
print(f"                     BaseCheck (execute lifecycle), CheckRunner (dependency resolution),")
print(f"                     LocalCI, PipelineManager, CIContext, BaseInferenceRunner,")
print(f"                     InferenceLogger, ThresholdStrategy, save_report, get_exit_code")
# ════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════
section("CLEANUP")
local_env.teardown()
cleanup(WORKSPACE)
print(f"✓ Workspace cleaned: {WORKSPACE}")
print(f"\n{'='*60}")
print(f"  🎉 ALL E2E TESTS PASSED 🎉")
print(f"{'='*60}")
