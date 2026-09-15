"""Validated YAML configuration for replaceable pipeline components.

The generic ``settings`` mapping exists only at the configuration boundary. Each adapter
must immediately validate it into its own typed settings model before construction; it is
never passed through the scientific pipeline as an unstructured dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import Field, JsonValue, model_validator

from oilspill.domain.common import ArtifactRef, FrozenModel, Identifier, TimeRange
from oilspill.domain.environment import EnvironmentalVariable
from oilspill.domain.geospatial import Duration, Length


class ComponentConfig(FrozenModel):
    """Selection and adapter-owned settings for one replaceable component."""

    name: Identifier
    config_file: str | None = None
    settings: dict[str, JsonValue] = Field(default_factory=dict)


class DetectorConfig(ComponentConfig):
    pass


class SARPreprocessorConfig(ComponentConfig):
    pass


class AISProviderConfig(ComponentConfig):
    pass


class EnvironmentalProviderConfig(ComponentConfig):
    pass


class SpillAgeEstimatorConfig(ComponentConfig):
    pass


class DriftModelConfig(ComponentConfig):
    pass


class AttributionModelConfig(ComponentConfig):
    pass


class PipelineConfig(FrozenModel):
    """Top-level component selections used by the composition root."""

    schema_version: int = Field(default=1, ge=1)
    detector: DetectorConfig
    sar_preprocessor: SARPreprocessorConfig
    ais_provider: AISProviderConfig
    environmental_provider: EnvironmentalProviderConfig
    spill_age_estimator: SpillAgeEstimatorConfig
    drift: DriftModelConfig
    attribution: AttributionModelConfig


class PipelineExecutionConfig(FrozenModel):
    """Non-scientific run controls and explicit source/output locations."""

    sar_source: ArtifactRef
    scene_id: Identifier | None = None
    output_path: Path
    random_seed: int
    environmental_variables: tuple[EnvironmentalVariable, ...] = Field(min_length=1)
    environmental_query_interval: TimeRange
    forecast_source_policy: Literal["highest_ranked_scored_candidate", "observed_slick"]
    forecast_initialization_validation_note: str | None = None
    pipeline_mode: Literal["legacy", "hybrid_mvp"] = "legacy"


class BackwardEvidenceConfig(FrozenModel):
    enabled: bool
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]


class PipelineSourceInferenceConfig(FrozenModel):
    release_enumerator: ComponentConfig
    observation_likelihood: ComponentConfig
    search: ComponentConfig
    backward_evidence: BackwardEvidenceConfig


class HybridAISConfig(FrozenModel):
    credible_region_percent: Literal[90, 95]
    spatial_buffer: Length
    temporal_margin_before: Duration
    temporal_margin_after: Duration
    validation_note: Identifier
    validation_status: Literal["DATASET_VALIDATION_REQUIRED"]

    @model_validator(mode="after")
    def _validated(self) -> HybridAISConfig:
        if "DATASET_VALIDATION_REQUIRED" not in self.validation_note:
            raise ValueError("hybrid AIS settings must mark DATASET_VALIDATION_REQUIRED")
        return self


class EndToEndPipelineConfig(PipelineConfig):
    """Complete composition specification consumed by ``PipelineOrchestrator``."""

    sar_reader: ComponentConfig
    spill_characterizer: ComponentConfig
    vessel_track_reconstructor: ComponentConfig
    candidate_generator: ComponentConfig
    release_hypothesis_generator: ComponentConfig | None = None
    attribution_evidence: ComponentConfig
    ensemble_forecaster: ComponentConfig
    coastline_provider: ComponentConfig
    coastal_impact: ComponentConfig
    execution: PipelineExecutionConfig
    source_inference: PipelineSourceInferenceConfig | None = None
    eulerian_transport: ComponentConfig | None = None
    hybrid_coarse_hindcast: ComponentConfig | None = None
    hybrid_hindcast: ComponentConfig | None = None
    hybrid_ais: HybridAISConfig | None = None
    hybrid_forecast: ComponentConfig | None = None

    @model_validator(mode="after")
    def _hybrid_components_present(self) -> EndToEndPipelineConfig:
        if self.execution.pipeline_mode == "hybrid_mvp":
            missing = tuple(
                name
                for name, component in (
                    ("eulerian_transport", self.eulerian_transport),
                    ("hybrid_coarse_hindcast", self.hybrid_coarse_hindcast),
                    ("hybrid_hindcast", self.hybrid_hindcast),
                    ("hybrid_ais", self.hybrid_ais),
                    ("hybrid_forecast", self.hybrid_forecast),
                )
                if component is None
            )
            if missing:
                raise ValueError(
                    "hybrid_mvp pipeline requires configured components: " + ", ".join(missing)
                )
        return self


ConfigT = TypeVar("ConfigT", bound=FrozenModel)


class ConfigurationError(ValueError):
    """Raised when a YAML document is unreadable or is not a mapping."""


def _merge_mappings(
    base: dict[str, object], overrides: dict[str, object]
) -> dict[str, object]:
    merged = dict(base)
    for key, value in overrides.items():
        prior = merged.get(key)
        if isinstance(prior, dict) and isinstance(value, dict):
            merged[key] = _merge_mappings(prior, value)
        else:
            merged[key] = value
    return merged


def _read_yaml_mapping(
    path: str | Path, *, _ancestors: tuple[Path, ...] = ()
) -> dict[str, object]:
    config_path = Path(path)
    resolved = config_path.resolve()
    if resolved in _ancestors:
        raise ConfigurationError(f"cyclic YAML configuration inheritance at {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError:
        # Preserve path-related exception types so presentation layers can distinguish an
        # invalid path from syntactically invalid configuration.
        raise
    except yaml.YAMLError as error:
        raise ConfigurationError(f"could not parse YAML configuration {config_path}") from error
    if not isinstance(raw, dict):
        raise ConfigurationError(f"YAML configuration {config_path} must contain a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ConfigurationError("YAML extends must be a non-empty relative path")
    parent_path = Path(parent)
    if parent_path.is_absolute():
        raise ConfigurationError("YAML extends must be relative to the child configuration")
    inherited = _read_yaml_mapping(
        config_path.parent / parent_path,
        _ancestors=(*_ancestors, resolved),
    )
    return _merge_mappings(inherited, raw)


def load_config(path: str | Path, model: type[ConfigT]) -> ConfigT:
    """Load one YAML mapping into the requested strict Pydantic configuration model."""

    return model.model_validate(_read_yaml_mapping(path))


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    return load_config(path, PipelineConfig)


def load_end_to_end_pipeline_config(path: str | Path) -> EndToEndPipelineConfig:
    return load_config(path, EndToEndPipelineConfig)


def load_component_config(path: str | Path, model: type[ConfigT]) -> ConfigT:
    return load_config(path, model)


class SourceInferenceConfig(FrozenModel):
    """Source-search composition, also used with the pipeline shared drift instance."""

    drift: ComponentConfig
    release_enumerator: ComponentConfig
    observation_likelihood: ComponentConfig
    search: ComponentConfig
