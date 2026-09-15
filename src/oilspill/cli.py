"""Thin command-line adapter over application services."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from oilspill.application import (
    CommandOutcome,
    benchmark_detection,
    benchmark_pipeline,
    benchmark_report,
    inspect_config,
    list_components,
    run_pipeline,
    validate_dataset,
)
from oilspill.config import ConfigurationError
from oilspill.errors import (
    ConfiguredArtifactError,
    MissingConfiguredArtifactError,
    OptionalDependencyError,
)
from oilspill.registry import UnknownComponentError

CommandHandler = Callable[[argparse.Namespace], CommandOutcome]


def _with_config(service: Callable[[str | Path], CommandOutcome]) -> CommandHandler:
    return lambda args: service(args.config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oilspill",
        description="Modular oil-spill research pipeline and benchmark tools.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="execute the configured end-to-end pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.set_defaults(handler=_with_config(run_pipeline))

    benchmark = commands.add_parser("benchmark", help="run or report reproducible benchmarks")
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    detection = benchmark_commands.add_parser("detection", help="benchmark detectors")
    detection.add_argument("--config", required=True, type=Path)
    detection.set_defaults(handler=_with_config(benchmark_detection))

    pipeline = benchmark_commands.add_parser("pipeline", help="benchmark the full pipeline")
    pipeline.add_argument("--config", required=True, type=Path)
    pipeline.set_defaults(handler=_with_config(benchmark_pipeline))

    report = benchmark_commands.add_parser("report", help="generate a benchmark report")
    report.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/report.yaml"),
        help="report configuration (default: configs/benchmarks/report.yaml)",
    )
    report.set_defaults(handler=_with_config(benchmark_report))

    inspect = commands.add_parser(
        "inspect-config", help="validate and print a resolved configuration"
    )
    inspect.add_argument("--config", required=True, type=Path)
    inspect.set_defaults(handler=_with_config(inspect_config))

    components = commands.add_parser("list-components", help="list registered replaceable adapters")
    components.set_defaults(handler=lambda _args: list_components())

    validate = commands.add_parser(
        "validate-dataset", help="validate a canonical local dataset manifest"
    )
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=_with_config(validate_dataset))
    return parser


def classify_cli_error(error: Exception) -> tuple[int, str]:
    """Map domain/composition failures to stable, actionable CLI categories."""

    if isinstance(error, MissingConfiguredArtifactError):
        return 4, "Missing model checkpoint"
    if isinstance(
        error,
        (
            OptionalDependencyError,
            ModuleNotFoundError,
            ImportError,
        ),
    ):
        return 3, "Missing external dependency"
    if isinstance(error, UnknownComponentError):
        return 5, "Unsupported adapter"
    if isinstance(error, (ConfigurationError, ValidationError, yaml.YAMLError)):
        return 2, "Invalid configuration"
    if isinstance(error, (FileNotFoundError, NotADirectoryError, IsADirectoryError)):
        return 6, "Invalid path"
    if isinstance(error, ConfiguredArtifactError):
        return 4, "Invalid model checkpoint"
    if isinstance(error, OSError):
        return 6, "Invalid path"
    return 1, "Command failed"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handler: CommandHandler = args.handler
    try:
        outcome = handler(args)
    except Exception as error:  # CLI boundary converts typed errors; no traceback for users.
        exit_code, label = classify_cli_error(error)
        print(f"{label}: {error}", file=sys.stderr)
        return exit_code
    print(outcome.stdout)
    for diagnostic in outcome.stderr:
        print(diagnostic, file=sys.stderr)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
