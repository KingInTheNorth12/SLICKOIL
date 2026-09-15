"""SNAP Graph Processing Tool backend isolated behind the SAR backend protocol."""

from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

from oilspill.adapters.sar.config import SARPreprocessingConfig, SnapGPTConfig
from oilspill.adapters.sar.errors import SARBackendExecutionError, SARDependencyError
from oilspill.adapters.sar.preprocessor import BackendExecutionResult
from oilspill.domain.common import ComponentMetadata, MetadataEntry


class SnapGPTBackend:
    """Execute an explicitly supplied, checksummed SNAP graph.

    No graph, operator sequence, band name, resolution, or Sentinel product layout is inferred.
    Those real-product decisions remain DATASET_VALIDATION_REQUIRED.
    """

    def __init__(self, config: SnapGPTConfig) -> None:
        self._config = config

    def execute(
        self,
        source: Path,
        destination: Path,
        config: SARPreprocessingConfig,
    ) -> BackendExecutionResult:
        executable = self._resolve_executable()
        self._validate_graph()
        if not source.is_file():
            raise SARBackendExecutionError(f"SNAP input does not exist: {source}")

        command = [
            str(executable),
            str(self._config.graph),
            f"-Pinput={source}",
            f"-Poutput={destination}",
            *(f"-P{item.key}={item.value}" for item in self._config.parameters),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise SARDependencyError(
                f"could not start SNAP gpt executable: {executable}"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no process output"
            raise SARBackendExecutionError(
                f"SNAP gpt failed with exit code {completed.returncode}: {detail}"
            )
        if not destination.is_file():
            raise SARBackendExecutionError(
                "SNAP gpt exited successfully but did not create the configured output raster"
            )
        executed = tuple(name for name, stage in config.stages() if stage.enabled)
        return BackendExecutionResult(output_path=destination, executed_stages=executed)

    def component_metadata(self) -> ComponentMetadata:
        return ComponentMetadata(
            name="snap_gpt",
            version="DATASET_VALIDATION_REQUIRED",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            framework="ESA SNAP Graph Processing Tool",
            configuration_sha256=sha256(self._config.model_dump_json().encode()).hexdigest(),
            attributes=(
                MetadataEntry(key="graph_sha256", value=self._config.graph_sha256),
                MetadataEntry(
                    key="graph_semantics_validation", value="DATASET_VALIDATION_REQUIRED"
                ),
            ),
        )

    def _resolve_executable(self) -> Path:
        configured = self._config.executable
        if configured.parent != Path("."):
            if not configured.is_file():
                raise SARDependencyError(
                    "SNAP gpt executable is unavailable at "
                    f"{configured}; install SNAP or configure "
                    "the executable path"
                )
            return configured
        resolved = shutil.which(str(configured))
        if resolved is None:
            raise SARDependencyError(
                f"SNAP gpt executable {configured!s} is unavailable; install ESA SNAP or configure "
                "an absolute executable path"
            )
        return Path(resolved)

    def _validate_graph(self) -> None:
        graph = self._config.graph
        if not graph.is_file():
            raise SARDependencyError(
                f"SNAP processing graph is unavailable at {graph}; provide a validated graph"
            )
        actual = sha256(graph.read_bytes()).hexdigest()
        if actual != self._config.graph_sha256:
            raise SARDependencyError(
                f"SNAP processing graph checksum mismatch for {graph}; refusing to execute"
            )
