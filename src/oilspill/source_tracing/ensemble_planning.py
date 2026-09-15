"""Serial, replayable candidate hypotheses; no trajectory interpolation or engine APIs."""

import random
from hashlib import sha256

import rasterio
from rasterio.warp import transform

from oilspill.domain.ais import CandidateDecision
from oilspill.domain.common import ComponentMetadata, MetadataEntry, TimeRange
from oilspill.domain.drift import ReleaseHypothesis
from oilspill.domain.geospatial import Coordinate, PointGeometry, SpatialGeometry
from oilspill.requests import ForwardTraceRequest, ReleaseGenerationRequest
from oilspill.source_tracing.ensemble_models import (
    CandidateCoverageIssue,
    SourceEnsembleConfig,
    SourceEnsemblePlan,
    SourceEnsembleRequest,
    SourceMemberPlan,
)
from oilspill.source_tracing.hypotheses import eligible_points


def plan_source_ensemble(
    config: SourceEnsembleConfig, root: SourceEnsembleRequest, drift: ComponentMetadata
) -> SourceEnsemblePlan:
    if drift.configuration_sha256 is None:
        raise ValueError("resolved drift configuration checksum is required")
    ids = [c.candidate_id for c in root.candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    if any(c.decision != CandidateDecision.RETAINED for c in root.candidates):
        raise ValueError("source ensembles accept retained candidates only")
    if any(c.observation_id != root.observation.observation_id for c in root.candidates):
        raise ValueError("candidate observation mismatch")
    supplemental = [e.candidate_id for e in root.supplemental_metrics]
    if len(set(supplemental)) != len(supplemental) or set(supplemental).difference(ids):
        raise ValueError("invalid supplemental candidate IDs")
    fields = {f.field_id: f for f in root.forcing}
    if len(fields) != len(root.forcing):
        raise ValueError("duplicate forcing IDs")
    identity = sha256(
        (root.model_dump_json() + config.model_dump_json() + drift.model_dump_json()).encode()
    ).hexdigest()
    interval = root.observation.discharge_time.interval
    members = []
    coverage = []
    for candidate in sorted(root.candidates, key=lambda c: c.candidate_id):
        # Candidate-local identity: independent of candidate ordering and other candidates' data.
        stream = sha256(
            (
                str(root.random_seed)
                + ":"
                + root.observation.model_dump_json()
                + ":"
                + candidate.model_dump_json()
                + ":"
                + config.model_dump_json()
                + ":"
                + drift.model_dump_json()
            ).encode()
        ).hexdigest()
        rng = random.Random(int(sha256((stream + ":sampling").encode()).hexdigest(), 16))
        seed_base = int(sha256((stream + ":engine").encode()).hexdigest()[:8], 16)
        issues = []
        if (
            interval.start < candidate.track.interval.start
            or interval.end > candidate.track.interval.end
        ):
            issues.append(
                "release interval outside usable track coverage; DATASET_VALIDATION_REQUIRED"
            )
        if any(
            g.start < interval.end and g.end > interval.start
            for g in candidate.track.coverage.detected_gaps
        ):
            issues.append("missing AIS sections in release interval; DATASET_VALIDATION_REQUIRED")
        coverage.append(
            CandidateCoverageIssue(candidate_id=candidate.candidate_id, issues=tuple(issues))
        )
        eligible = eligible_points(
            ReleaseGenerationRequest(observation=root.observation, candidate=candidate)
        )
        for index in range(config.member_count):
            timestamp = config.release_time.timestamp
            if config.release_time.kind == "uniform_existing_track_points":
                timestamp = rng.choice(eligible).observed_at if eligible else None
            sampled = []
            parameters: list[MetadataEntry] = []
            internal: list[MetadataEntry] = []
            offsets = {"offset_x_metres": 0.0, "offset_y_metres": 0.0}
            for u in config.uncertainties:
                d = u.distribution
                value = d.value if d.kind == "fixed" else rng.uniform(d.lower, d.upper)
                sampled.append(MetadataEntry(key=u.source_id, value=value))
                if u.parameter_key:
                    entry = MetadataEntry(key=u.parameter_key, value=value)
                    (internal if u.representation == "engine_internal" else parameters).append(
                        entry
                    )
                else:
                    offsets[u.target] = value
            forcing = (
                config.forcing.choices[0]
                if config.forcing.kind == "fixed"
                else rng.choice(config.forcing.choices)
            )
            engine_seed = (seed_base + index) % 2**32
            member_id = f"{stream}:member:{index}"
            request = None
            position = None
            error = None
            matches = tuple(p for p in eligible if p.observed_at == timestamp)
            if len(matches) != 1:
                error = (
                    "No unique canonical AIS point at sampled release time; "
                    "no interpolation/extrapolation; DATASET_VALIDATION_REQUIRED"
                )
            elif any(
                g.start < matches[0].observed_at < g.end
                for g in candidate.track.coverage.detected_gaps
            ):
                error = "Sample falls inside a declared AIS gap; DATASET_VALIDATION_REQUIRED"
            else:
                point = matches[0]
                position = point.position
                if offsets["offset_x_metres"] or offsets["offset_y_metres"]:
                    metric = rasterio.crs.CRS.from_user_input(config.distance.measurement_crs)
                    if not metric.is_projected or metric.linear_units not in {"metre", "meter"}:
                        raise ValueError(
                            "release offsets require an explicitly configured projected metre CRS"
                        )
                    source = position.crs.wkt or f"{position.crs.authority}:{position.crs.code}"
                    geometry = position.geometry
                    assert geometry.type == "Point"
                    xs, ys = transform(
                        source, metric, [geometry.coordinate.x], [geometry.coordinate.y]
                    )
                    xs, ys = transform(
                        metric,
                        source,
                        [xs[0] + offsets["offset_x_metres"]],
                        [ys[0] + offsets["offset_y_metres"]],
                    )
                    position = SpatialGeometry(
                        crs=position.crs,
                        geometry=PointGeometry(coordinate=Coordinate(x=xs[0], y=ys[0])),
                    )
                release = ReleaseHypothesis(
                    release_id=member_id,
                    geometry=position,
                    interval=TimeRange(start=point.observed_at, end=point.observed_at),
                    source_candidate_id=candidate.candidate_id,
                    assumptions=(
                        config.validation_note,
                        config.release_time.validation_note,
                        "Exact canonical AIS point; no interpolation/extrapolation; "
                        "DATASET_VALIDATION_REQUIRED",
                    ),
                )
                if set(forcing.field_ids).difference(fields) or (
                    config.require_forcing and not forcing.field_ids
                ):
                    error = "Missing configured forcing; DATASET_VALIDATION_REQUIRED"
                else:
                    request = ForwardTraceRequest(
                        observation=root.observation,
                        candidate=candidate,
                        release=release,
                        forcing=tuple(fields[k] for k in forcing.field_ids),
                        random_seed=engine_seed,
                        model_parameters=tuple(parameters),
                        configured_uncertainty=tuple(internal),
                    )
            members.append(
                SourceMemberPlan(
                    candidate_id=candidate.candidate_id,
                    member_id=member_id,
                    member_index=index,
                    engine_seed=engine_seed,
                    sampled_time=timestamp,
                    sampled_position=position,
                    forcing_member_id=forcing.member_id,
                    sampled_values=tuple(sampled),
                    request=request,
                    planning_error=error,
                )
            )
    # Finite 32-bit seed space: fail on collisions, never silently share streams.
    if len({m.engine_seed for m in members}) != len(members):
        raise ValueError("candidate engine seed collision; choose another root seed")
    return SourceEnsemblePlan(
        plan_id=identity,
        root=root,
        config=config,
        drift=drift,
        members=tuple(members),
        coverage_issues=tuple(coverage),
    )
