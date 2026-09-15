#!/usr/bin/env python3
"""Build or inspect frozen demo incidents and export their presentation index.

This is deliberately a provenance-first builder.  It downloads only the public
PANGAEA reference files; Sentinel, ERA5, Copernicus Marine and historical AIS
are never replaced with mock data when their authenticated source is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from demo_incident_stages import (
    coastline_verify,
    forcing_verify,
    run_detection,
    sar_prepare,
    sentinel_verify,
    source_files,
    synthetic_ais,
)

CASE_ID = "pangaea-ow-0491-20190705"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_INSTANCES_ROOT = PROJECT_ROOT / "demo/instances"
PRODUCT = "S1A_IW_GRDH_1SDV_20190705T154102_20190705T154127_027984_0328EF_8C32.SAFE"
CHECKPOINT = Path("runs/training/final-segmentation/checkpoints/best.pt")
CHECKPOINT_SHA256 = "aa5507cb63d56eac6dbeb82d58d9056b75698564d8447da6bbfb956e88adec51"
PANGAEA_BASE = "https://download.pangaea.de/dataset/980773/files"
NATURAL_EARTH_COASTLINE = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
ROI = {"west": 33.50, "south": 32.50, "east": 34.75, "north": 33.50}
PATCH_CORNERS = {
    "UL": [34.081261, 32.948495],
    "UR": [34.216286, 32.969647],
    "BR": [34.190346, 33.084969],
    "BL": [34.055321, 33.063817],
}
EXPECTED_BBOX = {"xmin": 80, "ymin": 257, "xmax": 480, "ymax": 416}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def available_case_directories(instances_root: Path = DEMO_INSTANCES_ROOT) -> list[str]:
    """Return all candidate case directories, including incomplete scaffolds."""
    if not instances_root.is_dir():
        return []
    return sorted(path.name for path in instances_root.iterdir() if path.is_dir())


def validate_case_root(
    parser: argparse.ArgumentParser, case_id: str, root: Path, instances_root: Path
) -> Path:
    """Require an explicitly selected, manifest-backed demo case before any work."""
    available = available_case_directories(instances_root)
    available_text = ", ".join(available) if available else "(none)"
    if not root.is_dir():
        parser.error(
            f"demo case '{case_id}' was not found at {root}. "
            f"Available case directories: {available_text}"
        )
    if not (root / "manifest.json").is_file():
        parser.error(
            f"demo case '{case_id}' is missing manifest.json at {root}. "
            f"Available case directories: {available_text}"
        )
    return root.resolve()


def detection_manifest_context(manifest: dict[str, Any], case_id: str) -> dict[str, str]:
    """Read the selected PANGAEA acquisition identity needed by detector provenance."""
    if manifest.get("case_id") != case_id:
        raise ValueError("manifest case_id does not match the selected case directory")
    patch = manifest.get("pangaea_patch")
    if not isinstance(patch, dict):
        raise ValueError("manifest pangaea_patch metadata is required for detection")
    required = ("acquisition_start_utc", "acquisition_end_utc")
    missing = [name for name in required if not isinstance(patch.get(name), str)]
    if missing:
        raise ValueError(f"manifest is missing detection metadata: {', '.join(missing)}")
    return {
        "case_id": case_id,
        "acquisition_start": patch["acquisition_start_utc"],
        "acquisition_end": patch["acquisition_end_utc"],
    }


def sar_manifest_context(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Read the selected case's Sentinel product and PANGAEA footprint unchanged."""
    if manifest.get("case_id") != case_id:
        raise ValueError("manifest case_id does not match the selected case directory")
    patch = manifest.get("pangaea_patch")
    if not isinstance(patch, dict):
        raise ValueError("manifest pangaea_patch metadata is required for SAR preparation")
    product = patch.get("sentinel_product")
    corners = patch.get("patch_corners_lon_lat")
    if not isinstance(product, str) or not product:
        raise ValueError("manifest is missing SAR metadata: sentinel_product")
    if not isinstance(corners, dict):
        raise ValueError("manifest is missing SAR metadata: patch_corners_lon_lat")
    values: list[tuple[float, float]] = []
    for name in ("ul", "ur", "br", "bl"):
        coordinate = corners.get(name)
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or not all(isinstance(value, int | float) for value in coordinate)
        ):
            raise ValueError(f"manifest SAR patch corner '{name}' must be [longitude, latitude]")
        values.append((float(coordinate[0]), float(coordinate[1])))
    return {"product": product, "patch": tuple(values)}


def inspect_manifest_backed_sar_case(root: Path, case_id: str, context: dict[str, Any]) -> None:
    """Report the selected case's Sentinel source without writing scientific artifacts."""
    verified = sentinel_verify(root, product=str(context["product"]))
    print(f"Case                 {case_id}")
    print(f"Sentinel product     {context['product']}")
    print(f"Sentinel source      {verified['status']}")
    if verified.get("reason"):
        print(f"Reason               {verified['reason']}")


def read_json(path: Path) -> dict[str, Any]:
    """Return an object payload, treating absent or malformed optional files as absent."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    return next((path for path in paths if path.is_file() and path.stat().st_size > 0), None)


def presentation_title(case_id: str, manifest: dict[str, Any]) -> str:
    """Use case metadata where available and a legible, deterministic fallback otherwise."""
    pangaea_match = re.fullmatch(r"pangaea-(ow-\d+)-\d{8}", case_id)
    if pangaea_match:
        return f"PANGAEA {pangaea_match.group(1).upper()}"
    source = manifest.get("source_dataset", {})
    if isinstance(source, dict) and source.get("name"):
        return str(source["name"])
    return case_id.replace("-", " ").upper()


def incident_label(title: str, observed_at: str | None) -> str:
    if not observed_at:
        return title
    try:
        parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        return title
    return f"{title} · {parsed.strftime('%d %b %Y')}"


def blocking_reason(readiness: dict[str, Any], stage: str) -> str:
    issues = [str(value) for value in readiness.get("blocking_issues", [])]
    keywords = {
        "hindcast": ("slick", "current", "OpenOil", "forcing"),
        "ais": ("source posterior", "AIS", "slick"),
        "forecast": ("slick", "current", "OpenOil", "forcing"),
        "coastal": ("coastline", "coast", "slick"),
    }
    for keyword in keywords.get(stage, ()):
        match = next((issue for issue in issues if keyword.lower() in issue.lower()), None)
        if match:
            return match
    status = readiness.get(f"{stage}_status") or readiness.get("attribution_status")
    if status:
        return str(status).replace("_", " ") + "."
    return "No frozen output is available for this stage."


def copy_presentation_asset(source: Path | None, target: Path, name: str) -> bool:
    destination = target / name
    if source is None:
        destination.unlink(missing_ok=True)
        return False
    if not destination.is_file() or destination.read_bytes() != source.read_bytes():
        shutil.copyfile(source, destination)
    return True


def build_presentation_incident(root: Path, target: Path) -> dict[str, Any]:
    """Export only small, browser-safe artifacts from one frozen incident."""
    manifest = read_json(root / "manifest.json")
    readiness = read_json(root / "readiness.json")
    preparation = read_json(root / "prepared/preparation.json")
    detection = read_json(root / "outputs/detection/result.json")
    case_id = str(manifest.get("case_id") or root.name)
    output = root / "outputs"
    presentation = read_json(output / "presentation.json")
    detection_dir = output / "detection"
    target.mkdir(parents=True, exist_ok=True)

    sources = {
        "vv.png": first_existing((detection_dir / "vv.png", root / "prepared/vv_preview.png")),
        "vh.png": first_existing((detection_dir / "vh.png", root / "prepared/vh_preview.png")),
        "probability.png": first_existing((detection_dir / "probability.png",)),
        "mask.png": first_existing((detection_dir / "mask.png",)),
        "overlay.png": first_existing((detection_dir / "overlay.png",)),
        "source-posterior.json": first_existing(
            (output / "hindcast/source-posterior.json", output / "source-posterior.json")
        ),
        "ais.json": first_existing((output / "attribution/ais.json", output / "ais.json")),
        "forecast.json": first_existing(
            (output / "forecast/forecast.json", output / "forecast.json")
        ),
        "coastline.json": first_existing(
            (output / "coastal/coastline.json", output / "coastline.json")
        ),
    }
    assets = {
        name.removesuffix(".png"): name
        for name, source in sources.items()
        if name.endswith(".png") and copy_presentation_asset(source, target, name)
    }
    for name, source in sources.items():
        if not name.endswith(".png"):
            copy_presentation_asset(source, target, name)
    artifacts = {
        key: filename
        for key, filename in {
            "source": "source-posterior.json",
            "ais": "ais.json",
            "forecast": "forecast.json",
            "coastal": "coastline.json",
        }.items()
        if sources[filename]
    }
    for stale in ("incident-demo.json", "result.json"):
        (target / stale).unlink(missing_ok=True)

    ais_metadata = next(
        (read_json(path) for path in sorted((root / "raw/ais").glob("*metadata*.json"))), {}
    )
    manifest_ais = manifest.get("ais", {})
    if not isinstance(manifest_ais, dict):
        manifest_ais = {}
    synthetic_ais = bool(
        ais_metadata.get("synthetic") or manifest_ais.get("source") == "synthetic"
    )
    source_ready = bool(readiness.get("hindcast_ready") and sources["source-posterior.json"])
    ais_ready = bool(readiness.get("attribution_ready") and sources["ais.json"])
    forecast_ready = bool(readiness.get("forecast_ready") and sources["forecast.json"])
    coastal_ready = bool(readiness.get("coastline_ready") and sources["coastline.json"])
    sentinel = manifest.get("sentinel", {}) if isinstance(manifest.get("sentinel"), dict) else {}
    checkpoint = (
        manifest.get("checkpoint", {})
        if isinstance(manifest.get("checkpoint"), dict)
        else {}
    )
    recorded = (
        checkpoint.get("recorded", {}) if isinstance(checkpoint.get("recorded"), dict) else {}
    )
    observed_at = str(manifest.get("observation_time") or "") or None
    title = presentation_title(case_id, manifest)
    incident = {
        "caseId": case_id,
        "title": title,
        "observedAt": observed_at,
        "source": "PANGAEA / Sentinel-1" if case_id.startswith("pangaea-") else "Frozen incident",
        "execution": "precomputed",
        "assets": assets,
        "artifacts": artifacts,
        "readiness": {
            "detection": bool(readiness.get("detection_ready") and detection),
            "hindcast": source_ready,
            "forecast": forecast_ready,
            "ais": bool(readiness.get("synthetic_ais_ready") or ais_ready),
            "aisSynthetic": synthetic_ais,
            "coastal": coastal_ready,
        },
        "blockedReasons": {
            "source": blocking_reason(readiness, "hindcast"),
            "ais": blocking_reason(readiness, "ais"),
            "forecast": blocking_reason(readiness, "forecast"),
            "coastal": blocking_reason(readiness, "coastal"),
        },
        "sar": {
            "platform": sentinel.get("platform"),
            "productId": sentinel.get("product") or preparation.get("source_product"),
            "acquiredAt": observed_at or (sentinel.get("start_time") or [None])[0],
            "polarizations": preparation.get("band_order") or detection.get("band_order"),
            "crs": preparation.get("crs") or detection.get("input_crs"),
            "dimensions": preparation.get("dimensions") or detection.get("input_dimensions"),
            "preprocessing": preparation.get("preprocessing_match")
            or detection.get("preprocessing_status"),
        },
        "detection": {
            "model": detection.get("model") or recorded.get("architecture"),
            "encoder": detection.get("encoder") or recorded.get("encoder"),
            "inputs": detection.get("band_order") or recorded.get("band_order"),
            "threshold": detection.get("threshold") or recorded.get("threshold"),
            "oilPixelCount": detection.get("oil_pixel_count"),
            "preprocessingMatch": detection.get("preprocessing_status")
            or preparation.get("preprocessing_match"),
            "execution": "Precomputed" if detection else None,
            "checkpoint": detection.get("checkpoint_sha256"),
        },
        "ais": {
            "source": "synthetic" if synthetic_ais else manifest_ais.get("source"),
            "label": ais_metadata.get("display_label") or manifest_ais.get("display_label"),
            "available": ais_ready,
        },
    }
    if presentation:
        incident["presentation"] = presentation
    write_json(target / "incident.json", incident)
    return {"caseId": case_id, "label": incident_label(title, observed_at), "incident": incident}


def export_demo_ui(instances_root: Path, ui_root: Path) -> list[dict[str, Any]]:
    """Index all frozen instances without exposing the filesystem to the browser."""
    target_root = ui_root / "data/instances"
    target_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for root in sorted(path for path in instances_root.iterdir() if path.is_dir()):
        manifest = read_json(root / "manifest.json")
        if not manifest or not (root / "readiness.json").is_file():
            continue
        exported = build_presentation_incident(root, target_root / root.name)
        incident = exported["incident"]
        readiness = incident["readiness"]
        entries.append(
            {
                "caseId": exported["caseId"],
                "label": exported["label"],
                "data": f"instances/{root.name}/incident.json",
                "status": {
                    "detection": "ready" if readiness["detection"] else "unavailable",
                    "hindcast": "ready" if readiness["hindcast"] else "blocked",
                    "ais": (
                        "synthetic"
                        if readiness["aisSynthetic"]
                        else ("ready" if readiness["ais"] else "unavailable")
                    ),
                    "forecast": "ready" if readiness["forecast"] else "blocked",
                    "coastal": "ready" if readiness["coastal"] else "blocked",
                },
            }
        )
    write_json(ui_root / "data/demo-index.json", {"generatedAt": utc_now(), "incidents": entries})
    return entries


def layout(root: Path) -> None:
    for relative in (
        "provenance",
        "raw/pangaea",
        "raw/sentinel",
        "raw/forcing/wind",
        "raw/forcing/currents",
        "raw/coastline",
        "raw/ais",
        "prepared",
        "outputs/detection",
        "outputs/hindcast",
        "outputs/attribution",
        "outputs/forecast",
        "outputs/coastal",
        "ui",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def fetch_public_pangaea(root: Path, commands: list[str]) -> None:
    destination = root / "raw/pangaea"
    for filename in ("ow-0491.jpg", "ow-0491.xml"):
        target = destination / filename
        url = f"{PANGAEA_BASE}/{filename}"
        if target.exists():
            commands.append(f"preserved existing download: {target.relative_to(root)}")
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        commands.append(f"GET {url}")
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                temporary.open("xb") as output,
            ):
                while block := response.read(65536):
                    output.write(block)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def jpeg_dimensions(path: Path) -> list[int]:
    """Read JPEG SOF dimensions without treating visual pixels as SAR input."""
    with path.open("rb") as source:
        if source.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG")
        while True:
            marker_prefix = source.read(1)
            while marker_prefix == b"\xff":
                marker = source.read(1)
                if marker not in {b"\xff", b"\x00"}:
                    break
                marker_prefix = source.read(1)
            else:
                raise ValueError("invalid JPEG marker stream")
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length = int.from_bytes(source.read(2), "big")
            if length < 2:
                raise ValueError("invalid JPEG segment length")
            if marker in {bytes([value]) for value in range(0xC0, 0xC4)} | {
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                data = source.read(5)
                return [int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")]
            source.seek(length - 2, 1)


def pangaea_validation(root: Path) -> dict[str, Any]:
    image, annotation = root / "raw/pangaea/ow-0491.jpg", root / "raw/pangaea/ow-0491.xml"
    result: dict[str, Any] = {
        "dataset_id": "PANGAEA.980773",
        "source_urls": {"jpg": f"{PANGAEA_BASE}/ow-0491.jpg", "xml": f"{PANGAEA_BASE}/ow-0491.xml"},
        "status": "MISSING",
    }
    if not (image.is_file() and annotation.is_file()):
        result["missing"] = [str(path.name) for path in (image, annotation) if not path.is_file()]
        return result
    image_size = jpeg_dimensions(image)
    from PIL import Image

    with Image.open(image) as opened:
        image_mode = opened.mode
        if [opened.width, opened.height] != image_size:
            raise ValueError("JPEG header and decoded dimensions disagree")
    root_xml = ET.parse(annotation).getroot()
    xml_filename = root_xml.findtext("filename")
    xml_size = [
        int(root_xml.findtext("size/width", "0")),
        int(root_xml.findtext("size/height", "0")),
    ]
    node = root_xml.find("object/bndbox")
    bbox = {key: int(node.findtext(key, "-1")) for key in EXPECTED_BBOX} if node is not None else {}
    result.update(
        {
            "status": "READY"
            if image_size == [640, 640] and xml_size == [640, 640] and bbox == EXPECTED_BBOX
            else "INVALID",
            "jpg_dimensions": image_size,
            "jpg_mode": image_mode,
            "xml_filename": xml_filename,
            "xml_dimensions": xml_size,
            "xml_bbox": bbox,
            "expected_bbox": EXPECTED_BBOX,
            "bbox_difference": {key: bbox[key] - value for key, value in EXPECTED_BBOX.items()}
            if bbox
            else None,
            "patch_identity": "S1_20190705_154012_154152_VV_2",
            "jpg_sha256": sha256(image),
            "xml_sha256": sha256(annotation),
        }
    )
    return result


def build_natural_earth_coastline(root: Path, commands: list[str]) -> None:
    """Preserve the official archive then derive only ROI-intersecting WGS84 lines."""
    coastline_dir = root / "raw/coastline"
    archive = coastline_dir / "ne_10m_coastline.zip"
    output = coastline_dir / "coastline.geojson"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        commands.append(f"GET {NATURAL_EARTH_COASTLINE}")
        try:
            with urllib.request.urlopen(NATURAL_EARTH_COASTLINE, timeout=60) as response:
                with temporary.open("xb") as stream:
                    while block := response.read(65536):
                        stream.write(block)
            temporary.replace(archive)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if output.exists():
        commands.append("preserved existing derived coastline.geojson")
        return
    with zipfile.ZipFile(archive) as bundle:
        shp_name = next(name for name in bundle.namelist() if name.endswith(".shp"))
        features = coastline_features(bundle.read(shp_name))
    write_json(
        output,
        {
            "type": "FeatureCollection",
            "name": "Natural Earth 10m coastline intersecting incident ROI",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": features,
        },
    )


def coastline_features(shapefile: bytes) -> list[dict[str, Any]]:
    """Minimal, read-only parser for standard ESRI PolyLine records in the NE archive."""
    west, south, east, north = ROI["west"], ROI["south"], ROI["east"], ROI["north"]
    offset, features = 100, []
    while offset < len(shapefile):
        _, length_words = struct.unpack(">2i", shapefile[offset : offset + 8])
        content = shapefile[offset + 8 : offset + 8 + 2 * length_words]
        offset += 8 + 2 * length_words
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            continue
        if shape_type != 3:
            raise ValueError(f"unsupported Natural Earth shape type {shape_type}")
        xmin, ymin, xmax, ymax = struct.unpack("<4d", content[4:36])
        if xmax < west or xmin > east or ymax < south or ymin > north:
            continue
        parts, points = struct.unpack("<2i", content[36:44])
        starts = struct.unpack(f"<{parts}i", content[44 : 44 + 4 * parts])
        coordinate_offset = 44 + 4 * parts
        coordinates = [
            struct.unpack(
                "<2d", content[coordinate_offset + 16 * i : coordinate_offset + 16 * i + 16]
            )
            for i in range(points)
        ]
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < parts else points
            line = [[x, y] for x, y in coordinates[start:end]]
            if len(line) >= 2:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "LineString", "coordinates": line},
                    }
                )
    return features


def checkpoint_validation() -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(CHECKPOINT), "expected_sha256": CHECKPOINT_SHA256}
    if not CHECKPOINT.is_file():
        return {**value, "status": "MISSING"}
    actual = sha256(CHECKPOINT)
    value.update(
        {"actual_sha256": actual, "status": "READY" if actual == CHECKPOINT_SHA256 else "INVALID"}
    )
    if actual != CHECKPOINT_SHA256:
        return value
    try:
        from oilspill.training.checkpoints import load_checkpoint

        payload = load_checkpoint(CHECKPOINT)
        model, raster = payload["model_config"], payload["raster_config"]
        value["recorded"] = {
            "architecture": model["architecture"],
            "encoder": model["encoder"],
            "in_channels": model["in_channels"],
            "classes": model["classes"],
            "scse_reduction": model["parameters"].get("scse_reduction"),
            "threshold": payload["threshold"],
            "normalization": raster["normalization"],
        }
    except Exception as error:  # The hash is still valuable when optional ML imports are absent.
        value["load_error"] = f"{type(error).__name__}: {error}"
    return value


def environment() -> dict[str, Any]:
    packages = (
        "numpy",
        "rasterio",
        "xarray",
        "h5netcdf",
        "h5py",
        "torch",
        "segmentation-models-pytorch",
        "copernicusmarine",
        "cdsapi",
        "opendrift",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    try:
        import rasterio

        versions["GDAL"] = rasterio.__gdal_version__
    except Exception:
        versions["GDAL"] = "unavailable"
    return {
        "recorded_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "SNAP": "not detected by this builder",
    }


def file_status(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def write_instance(root: Path, commands: list[str]) -> None:
    pangaea, checkpoint = pangaea_validation(root), checkpoint_validation()
    input_tif = root / "prepared/model_input_vv_vh.tif"
    sources = source_files(root)
    sentinel = sentinel_verify(root)
    forcing = forcing_verify(root)
    coastline = coastline_verify(root)
    ais = root / "raw/ais/synthetic_ais_20190702_20190705.csv"
    synthetic_metadata = root / "raw/ais/synthetic_ais_metadata.json"
    ais_ready = (
        ais.is_file()
        and synthetic_metadata.is_file()
        and json.loads(synthetic_metadata.read_text()).get("synthetic") is True
    )
    preparation = root / "prepared/preparation.json"
    input_ready = input_tif.is_file() and preparation.is_file()
    detection_ready = file_status(root / "outputs/detection/result.json")
    missing_ais = root / "raw/ais/README_MISSING_AIS.md"
    if not ais_ready and not missing_ais.exists():
        missing_ais.write_text(
            "# Historical AIS required\n\n"
            "No authorized historical AIS source was configured for this build. "
            "Supply real records only for 2019-07-02T15:40:12Z through "
            "2019-07-05T17:40:12Z over the incident ROI. Canonical fields: "
            "mmsi, timestamp (UTC), longitude, latitude, sog, cog, heading; "
            "vessel_name and vessel_type optional. "
            "Do not use the repository's 2024 AIS CSV for this incident.\n",
            encoding="utf-8",
        )
    manifest = {
        "case_id": CASE_ID,
        "observation_time": "2019-07-05T15:40:12Z",
        "created_or_updated_at": utc_now(),
        "pangaea": pangaea,
        "sentinel": {
            **sentinel,
            "reference_patch_start_vs_product_start": (
                "PANGAEA patch start 15:40:12Z precedes this product start "
                "15:41:02Z; publisher patch-to-product derivation is unverified"
            ),
        },
        "sar_input": {
            "path": "prepared/model_input_vv_vh.tif",
            "band_order": ["VV", "VH"],
            "status": "READY" if input_ready else "MISSING",
            "preprocessing_match": json.loads(preparation.read_text()).get("preprocessing_match")
            if input_ready
            else "not_assessed",
        },
        "wind": {"provider": "ERA5", **forcing["wind"], "requested_bbox": ROI},
        "currents": {"provider": "Copernicus Marine", **forcing["currents"], "requested_bbox": ROI},
        "coastline": {
            **coastline,
            "dataset": "Natural Earth 1:10m Physical Vectors, Coastline",
            "version": "10m download archive (version recorded by source filename)",
            "source": NATURAL_EARTH_COASTLINE,
            "crs": "EPSG:4326",
            "geometry_type": "LineString",
            "extraction_bbox": ROI,
        },
        "ais": {
            "path": str(ais),
            "status": "READY" if ais_ready else "MISSING",
            "source": "synthetic",
            "historical_observation": False,
            "display_label": "Synthetic AIS demonstration",
            "metadata": str(synthetic_metadata) if ais_ready else None,
        },
        "checkpoint": checkpoint,
        "validation_flags": [
            "DATASET_VALIDATION_REQUIRED",
            "raw_PANGAEA_JPEG_is_reference_only_not_model_input",
            "publisher_patch_pixel_correspondence_unverified",
            "PANGAEA_patch_start_precedes_named_SAFE_product_start",
        ],
        "patch": {
            "name": "S1_20190705_154012_154152_VV_2",
            "size_pixels": [640, 640],
            "corners_lon_lat": PATCH_CORNERS,
            "reference_slick_bbox_pixels": EXPECTED_BBOX,
        },
    }
    blocking_issues = []
    if sentinel["status"] != "READY":
        blocking_issues.append("Sentinel source product absent; VV/VH cannot be confirmed.")
    if forcing["wind"]["status"] != "READY":
        blocking_issues.append("ERA5 wind failed variable, space, or time validation.")
    if forcing["currents"]["status"] != "READY":
        blocking_issues.append(
            "July 2019 current file covers the patch but not the full requested ROI; "
            "the older July 2009 file is rejected."
        )
    if coastline["status"] != "READY":
        blocking_issues.append(
            "Natural Earth coastline does not intersect the specified working ROI."
        )
    if not ais_ready:
        blocking_issues.append("Synthetic AIS demonstration is not yet generated.")
    if not input_ready:
        blocking_issues.append("VV/VH model input has not passed preparation validation.")
    if not detection_ready:
        blocking_issues.append("Frozen detector output is not yet generated.")
    elif (
        json.loads((root / "outputs/detection/result.json").read_text()).get("oil_pixel_count", 0)
        == 0
    ):
        blocking_issues.append(
            "Detector produced no slick pixels at threshold 0.5; "
            "no observed slick geometry for source tracing."
        )
    blocking_issues.append(
        "OpenOil backend and forcing-reader compatibility are not configured or verified."
    )
    blocking_issues.append(
        "Existing xarray environmental adapter expects wind and current variables "
        "on one rectilinear source grid; these downloads use separate grids."
    )
    backend_ready = (
        False  # OpenOil is not installed/configured; reader compatibility is unverified.
    )
    readiness = {
        "sar_ready": sentinel["status"] == "READY",
        "model_input_ready": input_ready,
        "detection_ready": detection_ready,
        "detected_slick_ready": detection_ready
        and json.loads((root / "outputs/detection/result.json").read_text()).get(
            "oil_pixel_count", 0
        )
        > 0,
        "wind_ready": forcing["wind"]["status"] == "READY",
        "currents_ready": forcing["currents"]["status"] == "READY",
        "environment_adapter_ready": False,
        "coastline_ready": coastline["status"] == "READY",
        "synthetic_ais_ready": bool(ais_ready),
        "ais_ready": False,
        "hindcast_ready": backend_ready,
        "attribution_ready": False,
        "attribution_status": "synthetic_AIS_prepared_source_posterior_required"
        if ais_ready
        else "synthetic_AIS_missing",
        "forecast_ready": backend_ready,
        "blocking_issues": blocking_issues,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "readiness.json", readiness)
    inventory = []
    format_map = {
        "pangaea_jpg": "JPEG",
        "pangaea_xml": "VOC XML",
        "sentinel": "SAFE ZIP",
        "wind": "NetCDF/HDF5",
        "currents": "NetCDF/HDF5",
        "coastline": "GeoJSON",
        "ais": "CSV",
    }
    for kind, files in sources.items():
        for path in files:
            status = (
                sentinel["status"]
                if kind == "sentinel"
                else forcing[kind]["status"]
                if kind in ("wind", "currents")
                else coastline["status"]
                if kind == "coastline"
                else "READY"
                if kind == "ais" and ais_ready
                else pangaea["status"]
                if kind.startswith("pangaea")
                else "PRESENT"
            )
            if kind == "currents" and str(path) != forcing["currents"].get("path"):
                status = "REJECTED_2009"
            inventory.append(
                {
                    "type": kind.upper(),
                    "path": str(path),
                    "format": format_map.get(kind, path.suffix),
                    "status": status,
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    safe_dir = root / "raw/sentinel" / PRODUCT
    if safe_dir.is_dir():
        inventory.append(
            {
                "type": "SAR_EXTRACTED",
                "path": str(safe_dir),
                "format": "SAFE directory",
                "status": "READY",
            }
        )
    write_json(
        root / "provenance/inventory.json",
        {"case_id": CASE_ID, "recorded_at": utc_now(), "artifacts": inventory},
    )
    write_json(root / "provenance/environment.json", environment())
    write_json(
        root / "provenance/retrieval.json",
        {
            "recorded_at": utc_now(),
            "external_sources": [
                pangaea["source_urls"],
                {
                    "provider": "Copernicus Data Space",
                    "identifier": PRODUCT,
                    "source_url": "not recorded with user-supplied download",
                },
                {
                    "provider": "ERA5",
                    "variables": forcing["wind"].get("variables"),
                    "source_url": "not recorded with user-supplied download",
                },
                {
                    "provider": "Copernicus Marine",
                    "product_id": forcing["currents"]
                    .get("dataset_attributes", {})
                    .get("subset:productId"),
                    "dataset_id": forcing["currents"]
                    .get("dataset_attributes", {})
                    .get("subset:datasetId"),
                    "source_url": "not recorded with user-supplied download",
                },
            ],
            "credentials_not_persisted": True,
        },
    )
    write_json(
        root / "provenance/processing.json",
        {
            "recorded_at": utc_now(),
            "commands_executed": commands,
            "sar_processing": json.loads(preparation.read_text()).get("processing_stages")
            if input_ready
            else "NOT_RUN: SNAP graph unconfigured; source-specific calibration pending.",
            "inference": "completed through production adapter" if detection_ready else "NOT_RUN",
            "scientific_assumptions": [
                "PANGAEA JPEG is a visual reference only and is never passed "
                "to the two-channel detector.",
                "The Sentinel-to-Zenodo training preprocessing match is approximate.",
                "Synthetic AIS is demonstration data, not a historical observation.",
                "The first current NetCDF is dated July 2009 and is rejected; "
                "the July 2019 file covers the patch but not the full ROI.",
            ],
        },
    )
    write_json(
        root / "ui/incident-demo.json",
        {
            "case_id": CASE_ID,
            "manifest": "../manifest.json",
            "readiness": "../readiness.json",
            "execution": "precomputed",
            "ais": {
                "source": "synthetic",
                "historical": False,
                "display_label": "Synthetic AIS demonstration",
            },
            "status": "segmentation_ready" if detection_ready else "preparation_pending",
        },
    )
    checks = []
    important = [
        *sources["pangaea_jpg"],
        *sources["pangaea_xml"],
        *sources["sentinel"],
        *sources["wind"],
        *sources["currents"],
        *sources["coastline"],
        *sources["ais"],
        root / "raw/ais/synthetic_ais_metadata.json",
        root / "raw/coastline/ne_10m_coastline.zip",
    ]
    important.extend(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            "prepared" in path.relative_to(root).parts
            or "outputs" in path.relative_to(root).parts
            and "adapter" not in path.relative_to(root).parts
            or "provenance" in path.relative_to(root).parts
            or path.name in ("manifest.json", "readiness.json", "incident-demo.json")
        )
    )
    for path in sorted(set(important)):
        if path.is_file() and path.name != "checksums.sha256":
            checks.append(f"{sha256(path)}  {path.relative_to(root)}")
    (root / "provenance/checksums.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        help="name of an existing directory under demo/instances/",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["inspect", "pangaea", "sar", "forcing", "coastline", "ais", "detect", "ui", "all"],
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="explicit case root for controlled tests; defaults to demo/instances/<case>",
    )
    args = parser.parse_args()
    root = validate_case_root(
        parser,
        args.case,
        args.root if args.root is not None else DEMO_INSTANCES_ROOT / args.case,
        DEMO_INSTANCES_ROOT,
    )
    if args.stage == "ui":
        entries = export_demo_ui(DEMO_INSTANCES_ROOT, PROJECT_ROOT / "ui")
        print(f"Exported {len(entries)} frozen incident(s) to ui/data/demo-index.json")
        for entry in entries:
            print(
                f"{entry['caseId']:32} {entry['status']['detection']:11} "
                f"{entry['status']['ais']}"
            )
        return 0
    if args.case != CASE_ID and args.stage not in {"inspect", "sar", "detect"}:
        parser.error(
            f"scientific build stages are currently source-specific to '{CASE_ID}'; "
            f"'{args.case}' may only use --stage inspect, --stage sar, --stage detect, or --stage ui "
            "until a case-specific scientific builder is implemented."
        )
    if args.case != CASE_ID:
        try:
            manifest = read_json(root / "manifest.json")
            if args.stage == "inspect":
                inspect_manifest_backed_sar_case(root, args.case, sar_manifest_context(manifest, args.case))
                return 0
            if args.stage == "sar":
                context = sar_manifest_context(manifest, args.case)
                layout(root)
                sentinel_verify(root, extract=True, product=str(context["product"]))
                sar_prepare(root, **context)
                return 0
            context = detection_manifest_context(manifest, args.case)
        except ValueError as error:
            parser.error(str(error))
        layout(root)
        run_detection(root, CHECKPOINT, CHECKPOINT_SHA256, **context)
        return 0
    layout(root)
    commands = [" ".join(sys.argv)]
    previous = root / "provenance/processing.json"
    if previous.is_file():
        commands = [*json.loads(previous.read_text()).get("commands_executed", []), *commands]
    if args.stage in {"pangaea", "all"}:
        fetch_public_pangaea(root, commands)
    if args.stage in {"coastline", "all"}:
        build_natural_earth_coastline(root, commands)
    if args.stage in {"sar", "all"}:
        sentinel_verify(root, extract=True)
        sar_prepare(root)
    if args.stage in {"ais", "all"}:
        synthetic_ais(root)
    if args.stage in {"detect", "all"}:
        run_detection(root, CHECKPOINT, CHECKPOINT_SHA256)
    if args.stage in {"all", "detect"}:
        export_demo_ui(DEMO_INSTANCES_ROOT, PROJECT_ROOT / "ui")
    write_instance(root, commands)
    readiness = json.loads((root / "readiness.json").read_text(encoding="utf-8"))
    rows = [
        ("PANGAEA patch", pangaea_validation(root)["status"]),
        ("PANGAEA XML", pangaea_validation(root)["status"]),
        ("Sentinel-1 VV", sentinel_verify(root)["status"]),
        ("Sentinel-1 VH", sentinel_verify(root)["status"]),
        ("VV/VH model TIFF", "READY" if readiness["model_input_ready"] else "MISSING"),
        ("ERA5 wind", "READY" if readiness["wind_ready"] else "MISSING"),
        ("Ocean currents", "READY" if readiness["currents_ready"] else "BLOCKED"),
        ("Coastline", "READY" if readiness["coastline_ready"] else "BLOCKED"),
        ("Synthetic AIS", "READY" if readiness["synthetic_ais_ready"] else "MISSING"),
        ("Checkpoint", checkpoint_validation()["status"]),
        ("Detection output", "READY" if readiness["detection_ready"] else "MISSING"),
    ]
    for label, status in rows:
        print(f"{label:20} {status}")
    print("\nTYPE               PATH                                FORMAT       STATUS")
    for item in json.loads((root / "provenance/inventory.json").read_text())["artifacts"]:
        print(
            f"{item['type']:18} {Path(item['path']).relative_to(root)} "
            f"{item['format']:12} {item['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
