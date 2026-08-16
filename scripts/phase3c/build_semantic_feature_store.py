"""Render official LIBERO states and cache frozen DecisionNCE features.

The builder is intentionally split into two layers:

* camera/config helpers are dependency-free and are covered by CPU tests;
* the actual HDF5/LIBERO/DecisionNCE extraction is imported lazily and only
  runs on the SSH environment where those packages and data are installed.

No image key, orientation, preprocessing rule, or embedding dimension is
silently guessed.  A completed store therefore carries enough provenance to
reproduce the exact frame-to-feature mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import PHASE3C_SCHEMA_VERSION, assert_causal_input
from .io import load_json_config, write_json


FEATURE_STORE_SCHEMA = "phase3c-semantic-feature-store.v2"
DEFAULT_MODEL_ID = "DecisionNCE-P"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_path(value: str | os.PathLike[str]) -> Path:
    """Expand environment variables without treating an unknown variable as empty."""

    raw = os.fspath(value)
    expanded = os.path.expandvars(raw)
    if "$" in expanded or "%" in expanded:
        raise ValueError(f"unresolved environment variable in path: {raw}")
    return Path(expanded).expanduser()


def _jsonable_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def camera_inventory(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Describe observation entries that can be inspected as image candidates."""

    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    inventory: dict[str, dict[str, Any]] = {}
    for key in sorted(observation):
        value = observation[key]
        shape = _jsonable_shape(value)
        if shape is None or len(shape) not in (2, 3):
            continue
        dtype = getattr(value, "dtype", None)
        channels = shape[-1] if len(shape) == 3 else 1
        if channels not in (1, 3, 4) and shape[0] not in (1, 3, 4):
            continue
        inventory[str(key)] = {
            "dtype": str(dtype) if dtype is not None else type(value).__name__,
            "shape": shape,
            "channels_last_candidate": bool(len(shape) == 2 or channels in (1, 3, 4)),
        }
    return inventory


def _as_numpy(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    return np.asarray(image)


def normalize_camera_image(
    image: Any,
    *,
    channel_order: str = "rgb",
    vertical_flip: bool = False,
) -> np.ndarray:
    """Normalize one configured camera entry to finite HWC uint8 RGB.

    The orientation flags are explicit config values.  In particular, this
    function never flips an image merely because a shape looks unusual.
    """

    array = _as_numpy(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        pass
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"camera image must be HWC/CHW with 1/3/4 channels, got {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError(f"channel_order must be rgb or bgr, got {channel_order!r}")
    if channel_order == "bgr":
        array = array[..., ::-1]
    if vertical_flip:
        array = array[::-1, ...]
    if not np.isfinite(array).all():
        raise ValueError("camera image contains non-finite values")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    values = array.astype(np.float32, copy=False)
    if values.size and float(values.max()) <= 1.0 and float(values.min()) >= 0.0:
        values = values * 255.0
    if values.size and (float(values.min()) < 0.0 or float(values.max()) > 255.0):
        raise ValueError("camera image values are outside [0, 255]")
    return np.ascontiguousarray(np.rint(values).clip(0, 255).astype(np.uint8))


def configured_camera_frames(
    observation: Mapping[str, Any],
    camera_specs: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], dict[str, dict[str, Any]]]:
    """Select exactly the configured camera keys and return RGB frames."""

    if len(camera_specs) != 2:
        raise ValueError("Phase 3C requires exactly two configured camera specs")
    inventory = camera_inventory(observation)
    frames: list[np.ndarray] = []
    selected: dict[str, dict[str, Any]] = {}
    for index, raw_spec in enumerate(camera_specs):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"camera_specs[{index}] must be an object")
        key = str(raw_spec.get("key", ""))
        if not key or key not in observation:
            raise KeyError(f"configured camera key is absent: {key!r}; available={sorted(observation)}")
        frame = normalize_camera_image(
            observation[key],
            channel_order=str(raw_spec.get("channel_order", "rgb")),
            vertical_flip=bool(raw_spec.get("vertical_flip", False)),
        )
        alias = str(raw_spec.get("name", f"view{index}"))
        if alias in selected:
            raise ValueError(f"duplicate camera name: {alias}")
        selected[alias] = {
            "key": key,
            "name": alias,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "channel_order": str(raw_spec.get("channel_order", "rgb")),
            "vertical_flip": bool(raw_spec.get("vertical_flip", False)),
        }
        frames.append(frame)
    if len({tuple(frame.shape) for frame in frames}) != 1:
        raise ValueError(f"configured camera frames have different shapes: {[frame.shape for frame in frames]}")
    return frames, {"inventory": inventory, "selected": selected}


def frame_digest(image: Any) -> str:
    """Hash one normalized frame for orientation/determinism QA."""

    frame = normalize_camera_image(image)
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode("ascii"))
    digest.update(frame.tobytes(order="C"))
    return digest.hexdigest()


def required_frame_keys(joined_records: Iterable[Mapping[str, Any]]) -> dict[tuple[int, str], set[int]]:
    """Collect unique `(task_id, demo_key) -> steps` from the joined manifest."""

    required: dict[tuple[int, str], set[int]] = defaultdict(set)
    for index, record in enumerate(joined_records):
        try:
            task_id = int(record["task_id"])
            demo_key = str(record["demo_key"])
            steps = (int(record["prev_step"]), int(record["current_step"]), int(record["target_step"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"joined record {index} has invalid frame key fields") from exc
        if len(set(steps)) != 3:
            raise ValueError(f"joined record {index} must reference three distinct frame steps")
        required[(task_id, demo_key)].update(steps)
    return required


def _checkpoint_sha(path_value: Any) -> str | None:
    if not path_value:
        return None
    path = _expand_path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"DecisionNCE checkpoint does not exist: {path}")
    return sha256_file(path)


def _to_torch_batch(frames: Sequence[np.ndarray], *, preprocess: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - SSH-only path
        raise RuntimeError("DecisionNCE extraction requires torch") from exc
    arrays: list[np.ndarray] = []
    for frame in frames:
        normalized = normalize_camera_image(frame)
        array = normalized.astype(np.float32) / 255.0
        if preprocess == "rgb_01":
            pass
        elif preprocess == "rgb_-1_1":
            array = array * 2.0 - 1.0
        else:
            raise ValueError(f"unsupported explicit generic preprocessing: {preprocess!r}")
        arrays.append(np.transpose(array, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays, axis=0))


class DecisionNCEEncoder:
    """Small compatibility wrapper around the official DecisionNCE loader."""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        preprocess: str = "model",
        device: str | None = None,
        preprocess_fn: Any = None,
    ):
        import torch

        self.model = model
        self.model_id = str(model_id)
        self.preprocess = str(preprocess)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DecisionNCE config requested CUDA but CUDA is unavailable")
        self.model_preprocess = preprocess_fn or getattr(model, "preprocess", None)
        self._feature_dim: int | None = None
        if hasattr(model, "to"):
            model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "parameters"):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def load(cls, config: Mapping[str, Any]) -> "DecisionNCEEncoder":
        model_id = str(config.get("model_id", DEFAULT_MODEL_ID))
        module_name = str(config.get("python_module", "decisionnce"))
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - SSH-only path
            raise RuntimeError(
                f"cannot import DecisionNCE module {module_name!r}; install the pinned official repo"
            ) from exc
        loader = getattr(module, "DecisionNCE", None)
        if loader is None:
            loader = getattr(module, "load", None)
        if loader is None:
            raise RuntimeError(f"DecisionNCE loader is absent from module {module_name!r}")
        checkpoint = config.get("checkpoint")
        kwargs = dict(config.get("load_kwargs", {}) or {})
        if checkpoint:
            checkpoint_argument = str(config.get("checkpoint_argument", "checkpoint"))
            if not checkpoint_argument.isidentifier():
                raise ValueError("decisionnce.checkpoint_argument must be a Python identifier")
            kwargs[checkpoint_argument] = str(_expand_path(str(checkpoint)))
        if hasattr(loader, "load"):
            loaded = loader.load(model_id, **kwargs)
        elif callable(loader):
            loaded = loader(model_id, **kwargs)
        else:
            raise RuntimeError("configured DecisionNCE loader is not callable")
        preprocess_fn = None
        model = loaded
        if isinstance(loaded, (tuple, list)):
            if not loaded:
                raise RuntimeError("DecisionNCE loader returned an empty sequence")
            model = loaded[0]
            preprocess_fn = next((item for item in loaded[1:] if callable(item)), None)
        return cls(
            model,
            model_id=model_id,
            preprocess=str(config.get("preprocess", "model")),
            device=str(config.get("device")) if config.get("device") else None,
            preprocess_fn=preprocess_fn,
        )

    def _preprocess_images(self, frames: Sequence[np.ndarray]):
        model_preprocess = self.model_preprocess
        if callable(model_preprocess) and self.preprocess == "model":
            try:
                import torch
                from PIL import Image

                processed = [
                    model_preprocess(Image.fromarray(normalize_camera_image(frame)))
                    for frame in frames
                ]
                if all(torch.is_tensor(item) for item in processed):
                    return torch.stack(processed, dim=0).to(self.device)
            except Exception as exc:
                raise RuntimeError("DecisionNCE model preprocessing failed") from exc
            raise RuntimeError("DecisionNCE preprocess must return torch tensors")
        if self.preprocess == "model":
            raise RuntimeError("DecisionNCE model exposes no preprocess; set an explicit preprocess mode")
        return _to_torch_batch(frames, preprocess=self.preprocess).to(self.device)

    @staticmethod
    def _as_feature_tensor(value: Any):
        import torch

        if isinstance(value, (tuple, list)):
            value = value[0]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim > 2:
            value = value.flatten(start_dim=1)
        value = value.detach().float()
        if not torch.isfinite(value).all():
            raise ValueError("DecisionNCE emitted non-finite features")
        return value

    def encode_images(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        inputs = self._preprocess_images(frames)
        with torch.inference_mode():
            if hasattr(self.model, "encode_images"):
                output = self.model.encode_images(inputs)
            elif hasattr(self.model, "encode_image"):
                output = self.model.encode_image(inputs)
            elif callable(self.model):
                output = self.model(inputs)
            else:
                raise RuntimeError("DecisionNCE model has no image encoder method")
        features = self._as_feature_tensor(output).cpu().numpy().astype(np.float32)
        if features.shape[0] != len(frames):
            raise ValueError(f"DecisionNCE batch length mismatch: {features.shape} for {len(frames)} frames")
        if self._feature_dim is None:
            self._feature_dim = int(features.shape[1])
        elif int(features.shape[1]) != self._feature_dim:
            raise ValueError("DecisionNCE feature dimension changed within one store")
        return features

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if not hasattr(self.model, "encode_text") and not hasattr(self.model, "encode_texts"):
            raise RuntimeError("DecisionNCE model has no text encoder; language features are required")
        with torch.inference_mode():
            if hasattr(self.model, "encode_texts"):
                output = self.model.encode_texts(list(texts))
            else:
                output = self.model.encode_text(list(texts))
        features = self._as_feature_tensor(output).cpu().numpy().astype(np.float32)
        if features.shape[0] != len(texts):
            raise ValueError("DecisionNCE text batch length mismatch")
        if self._feature_dim is not None and int(features.shape[1]) != self._feature_dim:
            raise ValueError("image/text DecisionNCE dimensions do not match")
        self._feature_dim = int(features.shape[1])
        return features

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("feature_dim is unknown until an encode call")
        return self._feature_dim


def _hdf5_demo_group(handle: Any, demo_key: str) -> Any:
    data = handle.get("data")
    if data is None or demo_key not in data:
        raise KeyError(f"HDF5 demo group is absent: data/{demo_key}")
    group = data[demo_key]
    if "states" not in group:
        raise KeyError(f"HDF5 demo group has no states: data/{demo_key}")
    return group


def _resolve_hdf5(config: Mapping[str, Any], task_id: int) -> Path:
    mapping = config.get("hdf5_by_task")
    if not isinstance(mapping, Mapping):
        raise ValueError("semantic_feature_store.hdf5_by_task must be an object")
    value = mapping.get(str(task_id), mapping.get(task_id))
    if value is None:
        raise KeyError(f"no HDF5 path configured for task {task_id}")
    path = _expand_path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"configured HDF5 path does not exist: {path}")
    return path


def _resolve_language(config: Mapping[str, Any], task_id: int) -> str:
    mapping = config.get("task_languages")
    if isinstance(mapping, Mapping):
        value = mapping.get(str(task_id), mapping.get(task_id))
        if value:
            return str(value)
    suite_name = config.get("libero_suite")
    if not suite_name:
        raise ValueError(f"task_languages or libero_suite is required for task {task_id}")
    try:  # pragma: no cover - SSH-only path
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[str(suite_name)]()
        return str(suite.get_task(int(task_id)).language)
    except Exception as exc:  # pragma: no cover - SSH-only path
        raise RuntimeError(f"could not resolve LIBERO language for task {task_id}") from exc


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _iter_joined(path: Path) -> Iterable[dict[str, Any]]:
    from .io import iter_json_objects

    for index, item in enumerate(iter_json_objects(path)):
        record = dict(item)
        if record.get("schema") != PHASE3C_SCHEMA_VERSION:
            raise ValueError(f"joined record {index} has unsupported schema")
        assert_causal_input(record)
        yield record


def build_store(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build immutable per-demo `.npz` shards and a manifest."""

    store = dict(config.get("semantic_feature_store", config))
    joined_path = _expand_path(str(store["joined_manifest"]))
    output_root = _expand_path(str(store["output_root"]))
    camera_specs = store.get("cameras")
    if not isinstance(camera_specs, Sequence) or isinstance(camera_specs, (str, bytes)):
        raise ValueError("semantic_feature_store.cameras must contain exactly two entries")
    required = required_frame_keys(_iter_joined(joined_path))
    if not required:
        raise ValueError("joined manifest contains no frame keys")
    bddl_roots = [_expand_path(str(value)) for value in store.get("bddl_roots", [])]
    if not bddl_roots:
        raise ValueError("semantic_feature_store.bddl_roots must not be empty")
    if store.get("state_restore_tolerance") is None:
        raise ValueError(
            "semantic_feature_store.state_restore_tolerance must be frozen before extraction"
        )
    try:
        restore_tolerance = float(store["state_restore_tolerance"])
    except (TypeError, ValueError) as exc:
        raise ValueError("state_restore_tolerance must be a numeric frozen value") from exc
    if not np.isfinite(restore_tolerance) or restore_tolerance < 0.0:
        raise ValueError("state_restore_tolerance must be finite and non-negative")
    decision_config = dict(store.get("decisionnce", {}) or {})
    repository_commit = str(decision_config.get("repository_commit", "")).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", repository_commit):
        raise ValueError(
            "decisionnce.repository_commit must be an exact hexadecimal commit id"
        )
    if not decision_config.get("checkpoint"):
        raise ValueError("decisionnce.checkpoint is required for completed provenance")
    encoder = DecisionNCEEncoder.load(decision_config)
    model_checkpoint_sha = _checkpoint_sha(decision_config.get("checkpoint"))

    try:
        import h5py
        from scripts.phase2d.state_replay import _restore_observation, _state_error, make_environment
    except ImportError as exc:  # pragma: no cover - SSH-only path
        raise RuntimeError("semantic extraction requires h5py and LIBERO") from exc

    by_task_language: dict[int, str] = {}
    by_task_language_embedding: dict[int, np.ndarray] = {}
    hdf5_sha: dict[str, str] = {}
    bddl_by_task: dict[str, str] = {}
    index: dict[str, Any] = {}
    camera_inventory_records: dict[str, Any] = {}
    expected_camera_shape: tuple[int, int, int] | None = None
    completed = 0
    shard_sha256: dict[str, str] = {}
    for (task_id, demo_key), steps in sorted(required.items()):
        hdf5_path = _resolve_hdf5(store, task_id)
        by_task_language.setdefault(task_id, _resolve_language(store, task_id))
        if str(hdf5_path) not in hdf5_sha:
            hdf5_sha[str(hdf5_path)] = sha256_file(hdf5_path)
        environment, bddl_path = make_environment(
            hdf5_path,
            bddl_roots,
            camera_size=int(store.get("camera_size", 224)),
            render=True,
        )
        try:
            resolved_bddl = str(Path(bddl_path).resolve())
            previous_bddl = bddl_by_task.setdefault(str(task_id), resolved_bddl)
            if previous_bddl != resolved_bddl:
                raise ValueError(f"BDDL path changed within task {task_id}")
            with h5py.File(hdf5_path, "r") as handle:
                group = _hdf5_demo_group(handle, demo_key)
                states = group["states"]
                max_step = int(states.shape[0])
                invalid_steps = sorted(step for step in steps if step < 0 or step >= max_step)
                if invalid_steps:
                    raise IndexError(f"{demo_key} requested steps outside [0,{max_step}): {invalid_steps}")
                frame_batch: list[np.ndarray] = []
                restore_errors: list[float] = []
                for step in sorted(steps):
                    observation = _restore_observation(environment, states[step])
                    frames, inventory = configured_camera_frames(observation, camera_specs)
                    frame_shape = tuple(int(value) for value in frames[0].shape)
                    camera_size = int(store.get("camera_size", 224))
                    if frame_shape != (camera_size, camera_size, 3):
                        raise ValueError(
                            f"camera frame must be {(camera_size, camera_size, 3)}, got {frame_shape}"
                        )
                    if expected_camera_shape is None:
                        expected_camera_shape = frame_shape
                    elif frame_shape != expected_camera_shape:
                        raise ValueError(
                            f"camera frame shape changed from {expected_camera_shape} to {frame_shape}"
                        )
                    if not camera_inventory_records:
                        camera_inventory_records = inventory
                    elif camera_inventory_records.get("selected") != inventory.get("selected"):
                        raise ValueError("selected camera metadata changed during extraction")
                    frame_batch.extend(frames)
                    error = _state_error(environment, states[step])
                    restore_errors.append(float(error) if error is not None else float("nan"))
                encode_batch_size = int(store.get("encode_batch_size", 64))
                if encode_batch_size <= 0:
                    raise ValueError("encode_batch_size must be positive")
                encoded_frames = np.concatenate(
                    [
                        encoder.encode_images(
                            frame_batch[start : start + encode_batch_size]
                        )
                        for start in range(0, len(frame_batch), encode_batch_size)
                    ],
                    axis=0,
                )
                view_frames = [encoded_frames[0::2], encoded_frames[1::2]]
                if len(view_frames[0]) != len(steps) or len(view_frames[1]) != len(steps):
                    raise ValueError("DecisionNCE camera batching changed frame/view order")
                if task_id not in by_task_language_embedding:
                    by_task_language_embedding[task_id] = encoder.encode_texts(
                        [by_task_language[task_id]]
                    )[0]
                language = by_task_language_embedding[task_id]
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        relative = Path(str(task_id)) / f"{demo_key}.npz"
        shard_path = output_root / relative
        step_array = np.asarray(sorted(steps), dtype=np.int32)
        arrays = {
            "steps": step_array,
            "view0": np.asarray(view_frames[0], dtype=np.float32),
            "view1": np.asarray(view_frames[1], dtype=np.float32),
            "language": np.asarray(language, dtype=np.float32),
            "state_restore_max_abs": np.asarray(restore_errors, dtype=np.float64),
        }
        if (
            arrays["view0"].ndim != 2
            or arrays["view0"].shape[0] != len(step_array)
            or arrays["view1"].shape != arrays["view0"].shape
            or arrays["language"].shape != (arrays["view0"].shape[1],)
            or not np.isfinite(arrays["view0"]).all()
            or not np.isfinite(arrays["view1"]).all()
            or not np.isfinite(arrays["language"]).all()
        ):
            raise ValueError(f"invalid DecisionNCE shard shape for task={task_id} demo={demo_key}")
        if not np.isfinite(arrays["state_restore_max_abs"]).all():
            raise RuntimeError(f"simulator restore error was unavailable for task={task_id} demo={demo_key}")
        if float(arrays["state_restore_max_abs"].max(initial=0.0)) > restore_tolerance:
            raise RuntimeError(
                f"simulator restore error exceeded frozen tolerance for task={task_id} demo={demo_key}"
            )
        _atomic_npz(shard_path, arrays)
        shard_sha256[str(relative)] = sha256_file(shard_path)
        key_prefix = f"{task_id}/{demo_key}/"
        for row, step in enumerate(step_array.tolist()):
            index[f"{key_prefix}{step}/view0"] = {"shard": str(relative), "row": row, "view": 0}
            index[f"{key_prefix}{step}/view1"] = {"shard": str(relative), "row": row, "view": 1}
        index[f"{task_id}/{demo_key}/language"] = {"shard": str(relative), "row": 0, "view": "language"}
        completed += 1

    manifest = {
        "schema": FEATURE_STORE_SCHEMA,
        "phase3c_schema": PHASE3C_SCHEMA_VERSION,
        "status": "completed",
        "source": {"joined_manifest": str(joined_path), "joined_manifest_sha256": sha256_file(joined_path)},
        "decisionnce": {
            "model_id": str(decision_config.get("model_id", DEFAULT_MODEL_ID)),
            "python_module": str(decision_config.get("python_module", "decisionnce")),
            "repository_commit": repository_commit,
            "checkpoint": str(decision_config.get("checkpoint")) if decision_config.get("checkpoint") else None,
            "checkpoint_argument": str(decision_config.get("checkpoint_argument", "checkpoint")),
            "checkpoint_sha256": model_checkpoint_sha,
            "feature_dim": encoder.feature_dim,
            "preprocess": str(decision_config.get("preprocess", "model")),
            "device": str(encoder.device),
        },
        "camera": {"size": int(store.get("camera_size", 224)), "specs": [dict(item) for item in camera_specs], "inventory": camera_inventory_records},
        "task_languages": {str(key): value for key, value in sorted(by_task_language.items())},
        "hdf5_sha256": hdf5_sha,
        "bddl_by_task": bddl_by_task,
        "state_restore_tolerance": restore_tolerance,
        "index": index,
        "shard_sha256": shard_sha256,
        "shards": completed,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "qa" / "camera_inventory.json", camera_inventory_records)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI is exercised on SSH
    args = _parse_args()
    config = load_json_config(args.config)
    manifest = build_store(config)
    print(json.dumps({"status": manifest["status"], "shards": manifest["shards"], "feature_dim": manifest["decisionnce"]["feature_dim"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
