from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / "mineru.json"
BACKUP_SUFFIX = ".bak"
DEFAULT_MODEL_REPO = "opendatalab/MinerU2.5-2509-1.2B"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "mineru_models"


@dataclass
class VLLMSettings:
    """Settings that control the MinerU vLLM backend."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int = 16384
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    swap_space_mb: int = 8192

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "block_size": self.block_size,
            "swap_space_mb": self.swap_space_mb,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VLLMSettings":
        return cls(
            data_parallel_size=int(data.get("data_parallel_size", 1)),
            tensor_parallel_size=int(data.get("tensor_parallel_size", 1)),
            pipeline_parallel_size=int(data.get("pipeline_parallel_size", 1)),
            max_model_len=int(data.get("max_model_len", 16384)),
            trust_remote_code=bool(data.get("trust_remote_code", True)),
            dtype=str(data.get("dtype", "bfloat16")),
            gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.90)),
            block_size=int(data.get("block_size", 16)),
            swap_space_mb=int(data.get("swap_space_mb", 8192)),
        )


@dataclass
class MineruConfig:
    """Typed view over MinerU's JSON configuration."""

    model_path: Optional[Path] = None
    model_source: str = field(
        default_factory=lambda: os.environ.get("MINERU_MODEL_SOURCE", "huggingface")
    )
    backend_name: str = "vlm-vllm-engine"
    device: str = "cuda:0"
    cuda_visible_devices: str = "0"
    vllm_settings: VLLMSettings = field(default_factory=VLLMSettings)
    model_options: Dict[str, Any] = field(default_factory=dict)
    runtime_options: Dict[str, Any] = field(default_factory=dict)
    backend_options: Dict[str, Any] = field(default_factory=dict)
    additional: Dict[str, Any] = field(default_factory=dict)

    def set_model_path(self, path: Path) -> None:
        self.model_path = path.expanduser().resolve()

    def set_cuda_device(self, device_index: int) -> None:
        self.device = f"cuda:{device_index}"
        self.cuda_visible_devices = str(device_index)

    def set_visible_devices(self, devices: Sequence[int]) -> None:
        device_list = list(devices)
        if not device_list:
            raise ValueError("At least one CUDA device index must be provided.")
        self.cuda_visible_devices = ",".join(str(index) for index in device_list)
        self.device = f"cuda:{device_list[0]}"

    def update_vllm_settings(self, **settings: Any) -> None:
        for key, value in settings.items():
            if not hasattr(self.vllm_settings, key):
                raise AttributeError(f"Unknown vLLM setting '{key}'")
            setattr(self.vllm_settings, key, value)

    def to_dict(self) -> Dict[str, Any]:
        model_section: Dict[str, Any] = dict(self.model_options)
        if self.model_path:
            model_section["path"] = str(self.model_path)
        else:
            model_section.pop("path", None)
        model_section["source"] = self.model_source

        runtime_section: Dict[str, Any] = dict(self.runtime_options)
        runtime_section["device"] = self.device
        runtime_section["cuda_visible_devices"] = self.cuda_visible_devices

        backend_section: Dict[str, Any] = dict(self.backend_options)
        backend_section["name"] = self.backend_name
        backend_section["vllm"] = self.vllm_settings.to_dict()

        config: Dict[str, Any] = dict(self.additional)
        config["model"] = model_section
        config["runtime"] = runtime_section
        config["backend"] = backend_section
        return config


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Optional[Path] = None) -> MineruConfig:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        LOGGER.debug("No MinerU config found at %s; using defaults", config_path)
        return MineruConfig()

    raw_config = _read_json(config_path)

    model_section = raw_config.get("model", {})
    runtime_section = raw_config.get("runtime", {})
    backend_section = raw_config.get("backend", {})

    model_options = {
        key: value for key, value in model_section.items() if key not in {"path", "source"}
    }
    runtime_options = {
        key: value
        for key, value in runtime_section.items()
        if key not in {"device", "cuda_visible_devices"}
    }
    backend_options = {
        key: value for key, value in backend_section.items() if key not in {"name", "vllm"}
    }
    additional = {
        key: value
        for key, value in raw_config.items()
        if key not in {"model", "runtime", "backend"}
    }

    model_path_raw = model_section.get("path")
    model_path = Path(model_path_raw).expanduser() if isinstance(model_path_raw, str) else None
    if model_path is not None:
        model_path = model_path.resolve()

    default_source = os.environ.get("MINERU_MODEL_SOURCE", "huggingface")
    vllm_raw = backend_section.get("vllm") if isinstance(backend_section, Mapping) else {}
    vllm_mapping: Mapping[str, Any] = vllm_raw if isinstance(vllm_raw, Mapping) else {}

    config = MineruConfig(
        model_path=model_path,
        model_source=str(model_section.get("source", default_source)),
        backend_name=str(backend_section.get("name", "vlm-vllm-engine")),
        device=str(runtime_section.get("device", "cuda:0")),
        cuda_visible_devices=str(runtime_section.get("cuda_visible_devices", "0")),
        vllm_settings=VLLMSettings.from_dict(vllm_mapping),
        model_options=model_options,
        runtime_options=runtime_options,
        backend_options=backend_options,
        additional=additional,
    )
    return config


def _ensure_backup(path: Path, suffix: str = BACKUP_SUFFIX) -> Optional[Path]:
    if not path.exists():
        return None
    backup_path = path.with_suffix(path.suffix + suffix)
    shutil.copy2(path, backup_path)
    LOGGER.debug("Created MinerU config backup at %s", backup_path)
    return backup_path


def write_config(
    config: MineruConfig,
    path: Optional[Path] = None,
    *,
    make_backup: bool = True,
) -> Path:
    config_path = path or CONFIG_PATH
    if make_backup:
        _ensure_backup(config_path)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = config.to_dict()

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("MinerU configuration written to %s", config_path)
    return config_path


class ModelDownloadError(RuntimeError):
    """Raised when the MinerU model cannot be downloaded."""


def _download_from_huggingface(
    repo_id: str,
    cache_dir: Path,
    revision: Optional[str],
    token: Optional[str],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
        raise ModelDownloadError(
            "huggingface_hub is required to download models from HuggingFace. "
            "Install it with `uv pip install huggingface_hub`."
        ) from exc

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        token=token,
        local_files_only=False,
    )
    return Path(snapshot_path)


def _download_from_modelscope(
    model_id: str,
    cache_dir: Path,
    revision: Optional[str],
) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
        raise ModelDownloadError(
            "The modelscope package is required to download models from ModelScope. "
            "Install it with `uv pip install modelscope`."
        ) from exc

    kwargs: Dict[str, Any] = {"model_id": model_id, "cache_dir": str(cache_dir)}
    if revision is not None:
        kwargs["revision"] = revision
    snapshot_path = ms_snapshot_download(**kwargs)
    return Path(snapshot_path)


def ensure_model_downloaded(
    *,
    repo_id: str = DEFAULT_MODEL_REPO,
    cache_dir: Optional[Path] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    config: Optional[MineruConfig] = None,
) -> Path:
    """
    Download the MinerU model (if needed) and update the configuration.

    The hub is selected via the MINERU_MODEL_SOURCE environment variable or the
    value stored in the configuration (defaults to HuggingFace).
    """

    active_config = config or load_config()
    if active_config.model_path and active_config.model_path.exists():
        LOGGER.info("Using cached MinerU model at %s", active_config.model_path)
        return active_config.model_path

    model_source = os.environ.get("MINERU_MODEL_SOURCE", active_config.model_source).lower()
    repo_id = os.environ.get("MINERU_MODEL_REPO", repo_id)
    cache_root = cache_dir or DEFAULT_MODEL_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)

    if model_source == "huggingface":
        model_path = _download_from_huggingface(repo_id, cache_root, revision, token)
    elif model_source == "modelscope":
        model_path = _download_from_modelscope(repo_id, cache_root, revision)
    else:
        raise ModelDownloadError(
            f"Unsupported MINERU_MODEL_SOURCE '{model_source}'. Use 'huggingface' or 'modelscope'."
        )

    model_dir = model_path if model_path.is_dir() else model_path.parent
    active_config.set_model_path(model_dir)
    active_config.model_source = model_source
    if config is None:
        write_config(active_config)

    LOGGER.info("MinerU model available at %s", model_dir)
    return model_dir
