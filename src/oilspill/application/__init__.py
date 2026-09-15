"""Application services used by command-line and future UI composition layers."""

from oilspill.application.commands import (
    CommandOutcome,
    benchmark_detection,
    benchmark_pipeline,
    benchmark_report,
    inspect_config,
    list_components,
    run_pipeline,
    validate_dataset,
)

__all__ = [
    "CommandOutcome",
    "benchmark_detection",
    "benchmark_pipeline",
    "benchmark_report",
    "inspect_config",
    "list_components",
    "run_pipeline",
    "validate_dataset",
]
