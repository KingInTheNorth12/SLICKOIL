"""Use-case services for the CLI; argument parsing and science both live elsewhere."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from oilspill.application.dataset_validation import (
    load_dataset_validation_config,
    validate_canonical_detection_dataset,
)
from oilspill.benchmarking import (
    DetectionBenchmarkRunner,
    PipelineBenchmarkRunner,
    create_pipeline_metric_registry,
    generate_benchmark_report,
    load_benchmark_report_config,
    load_benchmarked_pipeline_config,
    load_detection_benchmark_config,
    load_pipeline_benchmark_config,
    load_pipeline_ground_truth,
    write_benchmark_report,
    write_detection_experiment,
    write_pipeline_benchmark_result,
)
from oilspill.benchmarking.models import load_detection_dataset
from oilspill.bootstrap import create_registries_with_builtins
from oilspill.config import (
    ComponentConfig,
    ConfigurationError,
    EndToEndPipelineConfig,
    load_config,
    load_end_to_end_pipeline_config,
)
from oilspill.pipeline import PipelineOrchestrator


class CommandOutcome(BaseModel):
    """Presentation-neutral command result."""

    stdout: str
    stderr: tuple[str, ...] = ()
    exit_code: int = 0


def run_pipeline(config_path: str | Path) -> CommandOutcome:
    config = load_end_to_end_pipeline_config(config_path)
    result = PipelineOrchestrator(config).run()
    payload = {
        "run_id": result.run_id,
        "status": "succeeded",
        "output_path": str(config.execution.output_path.resolve()),
    }
    return CommandOutcome(stdout=_json(payload))


def benchmark_detection(config_path: str | Path) -> CommandOutcome:
    config = load_detection_benchmark_config(config_path)
    dataset = load_detection_dataset(config.dataset.manifest)
    registries = create_registries_with_builtins()
    result = DetectionBenchmarkRunner(config, registries.detector).run(dataset)
    artifacts = write_detection_experiment(result, config.output_directory)
    failures = tuple(model for model in result.models if model.status == "failed")
    diagnostics = tuple(
        f"{_model_failure_label(model.error_type)} [{model.model_name}]: "
        f"{model.error_message or 'no error detail was provided'}"
        for model in failures
    )
    return CommandOutcome(
        stdout=str(artifacts.experiment_directory),
        stderr=diagnostics,
        exit_code=1 if failures else 0,
    )


def benchmark_pipeline(config_path: str | Path) -> CommandOutcome:
    benchmark_config = load_pipeline_benchmark_config(config_path)
    pipeline_config = load_benchmarked_pipeline_config(benchmark_config)
    ground_truth = load_pipeline_ground_truth(benchmark_config.ground_truth_manifest)
    result = PipelineBenchmarkRunner(
        benchmark_config,
        pipeline_config,
        ground_truth,
        create_pipeline_metric_registry(),
    ).run()
    directory = write_pipeline_benchmark_result(result, benchmark_config.output_directory)
    return CommandOutcome(stdout=str(directory))


def benchmark_report(config_path: str | Path) -> CommandOutcome:
    config = load_benchmark_report_config(config_path)
    summary = generate_benchmark_report(config)
    artifacts = write_benchmark_report(config, summary)
    return CommandOutcome(stdout=str(artifacts.report_directory))


def inspect_config(config_path: str | Path) -> CommandOutcome:
    """Validate a known configuration shape and print its fully resolved representation."""

    path = Path(config_path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        raise
    except yaml.YAMLError as error:
        raise ConfigurationError(f"could not parse YAML configuration {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError(f"YAML configuration {path} must contain a mapping")

    loader = _select_config_loader(raw)
    config = loader(path)
    return CommandOutcome(stdout=config.model_dump_json(indent=2))


def list_components() -> CommandOutcome:
    registries = create_registries_with_builtins()
    catalog = {
        field_name: list(getattr(registries, field_name).available())
        for field_name in registries.__dataclass_fields__
    }
    return CommandOutcome(stdout=_json({"schema_version": 1, "components": catalog}))


def validate_dataset(config_path: str | Path) -> CommandOutcome:
    config = load_dataset_validation_config(config_path)
    result = validate_canonical_detection_dataset(config)
    return CommandOutcome(
        stdout=result.model_dump_json(indent=2), exit_code=0 if result.valid else 1
    )


def _select_config_loader(raw: dict[object, object]) -> Callable[[Path], BaseModel]:
    keys = {key for key in raw if isinstance(key, str)}
    if "result_directories" in keys:
        return lambda path: load_benchmark_report_config(path)
    if {"pipeline_config", "ground_truth_manifest"} <= keys:
        return lambda path: load_pipeline_benchmark_config(path)
    if {"dataset", "models", "preprocessing"} <= keys:
        return lambda path: load_detection_benchmark_config(path)
    if "execution" in keys:
        return lambda path: load_config(path, EndToEndPipelineConfig)
    if "manifest" in keys and keys <= {
        "schema_version",
        "manifest",
        "verify_checksums",
        "verify_byte_sizes",
    }:
        return lambda path: load_dataset_validation_config(path)
    if "name" in keys:
        return lambda path: load_config(path, ComponentConfig)
    raise ConfigurationError(
        "unrecognized configuration shape; expected pipeline, benchmark, report, "
        "dataset-validation, or component configuration"
    )


def _model_failure_label(error_type: str | None) -> str:
    if error_type == "MissingModelCheckpointError":
        return "Missing model checkpoint"
    if error_type and "DependencyError" in error_type:
        return "Missing external dependency"
    if error_type == "UnknownComponentError":
        return "Unsupported adapter"
    return "Model benchmark failed"


def _json(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, default=str)
