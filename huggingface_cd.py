"""
Demo-only HuggingFaceCD for external run_e2e.py scripts.
Do not place in MLForge source unless you want to productize it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from mlforge.core.deployment_manager.basecd import BaseCD
from mlforge.core.deployment_manager.cd.deploy.base import BaseDeployer
from mlforge.core.deployment_manager.cd.gate import DeploymentGate
from mlforge.core.deployment_manager.cd.results import CDResult
from mlforge.core.deployment_manager.cd.strategies import BaseCDStrategy
from mlforge.core.deployment_manager.streaming.events import CDEvent
from mlforge.utils.logger import logger


class HuggingFaceCD(BaseCD):
    """
    Concrete BaseCD subclass for HuggingFace deployments.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        **BaseCD.DEFAULT_CONFIG,
        "hf_repo_id": None,
        "hf_token": None,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        gate: DeploymentGate,
        strategy: BaseCDStrategy,
        deployer: BaseDeployer,
        emit: Optional[Callable[[CDEvent], None]] = None,
    ) -> None:
        # IMPORTANT:
        # BaseDeploymentManager.__init__ calls resolve_conflicts() before BaseCD
        # assigns self.deployer. So do deployer-dependent checks HERE first.
        cfg = dict(config or {})

        repo_id = cfg.get("hf_repo_id") or getattr(deployer, "_repo_id", None)
        token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or getattr(deployer, "_token", None)

        if not repo_id:
            raise RuntimeError(
                "HuggingFaceCD: missing HuggingFace repo id. "
                "Pass repo_id to HuggingFaceDeployer or set config['hf_repo_id']."
            )
        if not token:
            raise RuntimeError(
                "HuggingFaceCD: no HuggingFace token found. "
                "Set HF_TOKEN or pass token to HuggingFaceDeployer."
            )

        cfg.setdefault("hf_repo_id", repo_id)
        cfg.setdefault("hf_token", token)

        super().__init__(cfg, gate, strategy, deployer, emit)

    def resolve_conflicts(self) -> None:
        try:
            import huggingface_hub  
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFaceCD: huggingface_hub is not installed. "
                "Run: pip install huggingface_hub"
            ) from exc

        model_path = self.config.get("model_path")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"HuggingFaceCD: model_path must be an existing directory: {model_path}"
            )

        logger.info(
            "HuggingFaceCD prerequisites validated: "
            f"repo='{self.config.get('hf_repo_id')}'"
        )

    def _pre_deploy_hook(self, ci_result) -> None:
        self.deployer.initialize()

    def _post_deploy_hook(self, cd_result: Optional[CDResult]) -> None:
        self.deployer.teardown()