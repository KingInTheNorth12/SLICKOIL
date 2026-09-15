"""Source-specific preparation for the frozen OW-0491 incident.

Only source-format knowledge lives here. Detection still uses the repository ports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

CASE = "pangaea-ow-0491-20190705"
PRODUCT = "S1A_IW_GRDH_1SDV_20190705T154102_20190705T154127_027984_0328EF_8C32.SAFE"
START = np.datetime64("2019-07-02T15:40:12")
END = np.datetime64("2019-07-07T15:40:12")
ROI = (33.50, 32.50, 34.75, 33.50)
PATCH = (
    (34.081261, 32.948495),
    (34.216286, 32.969647),
    (34.190346, 33.084969),
    (34.055321, 33.063817),
)


def source_files(root: Path) -> dict[str, list[Path]]:
    """Discover by format and source metadata; filenames are only candidates."""
    raw = root / "raw"
    found: dict[str, list[Path]] = {
        k: []
        for k in (
            "pangaea_jpg",
            "pangaea_xml",
            "sentinel",
            "wind",
            "currents",
            "coastline",
            "ais",
            "other",
        )
    }
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = path.relative_to(raw).parts
        if path.suffix.lower() == ".jpg" and "pangaea" in relative:
            kind = "pangaea_jpg"
        elif path.suffix.lower() == ".xml" and "pangaea" in relative:
            kind = "pangaea_xml"
        elif path.suffix.lower() == ".zip" and "sentinel" in relative:
            kind = "sentinel"
        elif path.suffix.lower() == ".nc" and "forcing" in relative:
            kind = "wind" if "wind" in relative else "currents"
        elif path.suffix.lower() == ".geojson" and "coastline" in relative:
            kind = "coastline"
        elif path.suffix.lower() == ".csv" and "ais" in relative:
            kind = "ais"
        else:
            kind = "other"
        found[kind].append(path)
    return found


def _sentinel_product_times(product: str) -> tuple[str, str]:
    """Return the acquisition bounds encoded in a Sentinel-1 SAFE product name."""
    parts = product.removesuffix(".SAFE").split("_")
    if len(parts) < 6:
        raise ValueError("Sentinel product name does not contain acquisition times")
    try:
        start = datetime.strptime(parts[4], "%Y%m%dT%H%M%S").isoformat()
        end = datetime.strptime(parts[5], "%Y%m%dT%H%M%S").isoformat()
    except ValueError as error:
        raise ValueError("Sentinel product name contains invalid acquisition times") from error
    return start, end


def sentinel_verify(
    root: Path, *, extract: bool = False, product: str = PRODUCT
) -> dict[str, Any]:
    import rasterio

    expected_start, expected_end = _sentinel_product_times(product)
    archives = source_files(root)["sentinel"]
    if len(archives) != 1:
        return {"status": "MISSING", "reason": "exactly one Sentinel ZIP is required"}
    archive = archives[0]
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        manifests = [n for n in names if n.endswith("/manifest.safe")]
        if len(manifests) != 1 or manifests[0].split("/")[0] != product:
            return {"status": "INVALID", "reason": "ZIP product identity mismatch"}
        manifest = ET.fromstring(bundle.read(manifests[0]))
        values: dict[str, list[str]] = {}
        for element in manifest.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if element.text and element.text.strip():
                values.setdefault(tag, []).append(element.text.strip())
        measurements = {
            pol: [
                n
                for n in names
                if "/measurement/" in n and n.endswith(".tiff") and f"-{pol.lower()}-" in n
            ]
            for pol in ("VV", "VH")
        }
        calibrations = {
            pol: [
                n
                for n in names
                if "/annotation/calibration/calibration-" in n
                and f"-{pol.lower()}-" in n
                and n.endswith(".xml")
            ]
            for pol in ("VV", "VH")
        }
        noise = {
            pol: [
                n
                for n in names
                if "/annotation/calibration/noise-" in n
                and f"-{pol.lower()}-" in n
                and n.endswith(".xml")
            ]
            for pol in ("VV", "VH")
        }
        annotations = {
            pol: [
                n
                for n in names
                if "/annotation/" in n
                and "/calibration/" not in n
                and f"-{pol.lower()}-" in n
                and n.endswith(".xml")
            ]
            for pol in ("VV", "VH")
        }
        identity = (
            "A" in values.get("number", [])
            and "IW" in values.get("mode", [])
            and "GRD" in values.get("productType", [])
            and any(v.startswith(expected_start) for v in values.get("startTime", []))
            and any(v.startswith(expected_end) for v in values.get("stopTime", []))
        )
        complete = identity and all(
            len(mapping[p]) == 1
            for mapping in (measurements, calibrations, noise, annotations)
            for p in ("VV", "VH")
        )
        details: dict[str, Any] = {
            "status": "READY" if complete else "INVALID",
            "archive": str(archive),
            "product": product,
            "platform": "S1A",
            "mode": "IW",
            "product_type": "GRD",
            "start_time": values.get("startTime"),
            "stop_time": values.get("stopTime"),
            "orbit_number": values.get("orbitNumber"),
            "relative_orbit_number": values.get("relativeOrbitNumber"),
            "pass": values.get("pass"),
            "measurement": measurements,
            "calibration": calibrations,
            "noise": noise,
            "geolocation_annotation": annotations,
        }
        if not complete:
            return details
        for pol in ("VV", "VH"):
            path = f"/vsizip/{archive.resolve()}/{measurements[pol][0]}"
            with rasterio.open(path) as raster:
                gcps, crs = raster.gcps
                if raster.count != 1 or len(gcps) < 4 or crs is None:
                    details["status"] = "INVALID"
                    details["reason"] = f"{pol} measurement lacks usable geolocation GCPs"
                    return details
                details[f"{pol.lower()}_raster"] = {
                    "width": raster.width,
                    "height": raster.height,
                    "dtype": raster.dtypes[0],
                    "gcp_count": len(gcps),
                    "gcp_crs": str(crs),
                }
        if extract:
            target = root / "raw/sentinel" / product
            if not (target / "manifest.safe").is_file():
                for item in bundle.infolist():
                    relative = Path(item.filename)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or relative.parts[0] != product
                    ):
                        raise ValueError("unsafe ZIP member path")
                    destination = archive.parent / relative
                    if item.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        continue
                    with bundle.open(item) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output)
            details["extracted_safe"] = str(target)
        return details


def forcing_verify(root: Path) -> dict[str, dict[str, Any]]:
    import xarray as xr

    result: dict[str, dict[str, Any]] = {}
    for kind in ("wind", "currents"):
        files = source_files(root)[kind]
        if not files:
            result[kind] = {"status": "MISSING", "reason": "source NetCDF absent"}
            continue
        candidates = []
        for candidate in files:
            with xr.open_dataset(candidate, engine="h5netcdf") as dataset:
                time_name = "valid_time" if "valid_time" in dataset.coords else "time"
                times = dataset[time_name].values.astype("datetime64[ns]")
                candidates.append(
                    {
                        "path": str(candidate),
                        "time_start": str(times[0]),
                        "time_end": str(times[-1]),
                        "time_coverage_valid": bool(times[0] <= START and times[-1] >= END),
                    }
                )
        matching = [item for item in candidates if item["time_coverage_valid"]]
        if len(matching) != 1:
            result[kind] = {
                "status": "INVALID",
                "reason": "no unique July 2019 source NetCDF",
                "candidates": candidates,
            }
            continue
        path = Path(str(matching[0]["path"]))
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            coordinate = "valid_time" if "valid_time" in dataset.coords else "time"
            values = dataset[coordinate].values.astype("datetime64[ns]")
            latitude = np.asarray(dataset.latitude.values)
            longitude = np.asarray(dataset.longitude.values)
            variable_names = ("u10", "v10") if kind == "wind" else ("uo", "vo")
            expected_names = (
                ("10 metre U wind component", "10 metre V wind component")
                if kind == "wind"
                else ("eastward_sea_water_velocity", "northward_sea_water_velocity")
            )
            matched = all(
                name in dataset
                and expected_names[index].lower()
                in str(
                    dataset[name].attrs.get("long_name" if kind == "wind" else "standard_name", "")
                ).lower()
                for index, name in enumerate(variable_names)
            )
            finite = {
                name: int(np.isfinite(dataset[name].values).sum())
                for name in variable_names
                if name in dataset
            }
            has_depth = "depth" in dataset.coords or "depth" in dataset.dims
            time_ready = len(values) > 1 and values[0] <= START and values[-1] >= END
            area_ready = (
                latitude.min() <= ROI[1]
                and latitude.max() >= ROI[3]
                and longitude.min() <= ROI[0]
                and longitude.max() >= ROI[2]
            )
            lon_half = float(np.median(np.abs(np.diff(longitude)))) / 2
            lat_half = float(np.median(np.abs(np.diff(latitude)))) / 2
            cell_extent = [
                float(longitude.min()) - lon_half,
                float(latitude.min()) - lat_half,
                float(longitude.max()) + lon_half,
                float(latitude.max()) + lat_half,
            ]
            finite_ready = all(finite.get(name, 0) > 0 for name in variable_names)
            units_ready = all(
                dataset[name].attrs.get("units") in ("m s-1", "m s**-1")
                for name in variable_names
                if name in dataset
            )
            time_axis_ready = len(values) > 1 and bool(
                np.all(np.diff(values) > np.timedelta64(0, "ns"))
            )
            patch_area_ready = (
                latitude.min() <= min(point[1] for point in PATCH)
                and latitude.max() >= max(point[1] for point in PATCH)
                and longitude.min() <= min(point[0] for point in PATCH)
                and longitude.max() >= max(point[0] for point in PATCH)
            )
            core_ready = matched and time_ready and finite_ready and units_ready and time_axis_ready
            status = (
                "READY"
                if core_ready and area_ready
                else "PARTIAL_ROI"
                if core_ready and patch_area_ready
                else "INVALID"
            )
            result[kind] = {
                "status": status,
                "path": str(path),
                "candidates": candidates,
                "variables": list(variable_names),
                "units": {
                    name: dataset[name].attrs.get("units")
                    for name in variable_names
                    if name in dataset
                },
                "time_name": coordinate,
                "time_start": str(values[0]),
                "time_end": str(values[-1]),
                "time_count": len(values),
                "latitude_start_end": [float(latitude[0]), float(latitude[-1])],
                "longitude_start_end": [float(longitude[0]), float(longitude[-1])],
                "approximate_cell_edge_extent": cell_extent,
                "spatial_resolution_degrees": [
                    float(np.median(np.abs(np.diff(longitude)))),
                    float(np.median(np.abs(np.diff(latitude)))),
                ],
                "temporal_resolution": str(values[1] - values[0]),
                "finite_cells": finite,
                "fill_values": {
                    name: dataset[name].attrs.get(
                        "_FillValue", dataset[name].encoding.get("_FillValue")
                    )
                    for name in variable_names
                    if name in dataset
                },
                "dataset_attributes": {
                    key: str(value)
                    for key, value in dataset.attrs.items()
                    if key.startswith("subset:")
                },
                "depth_coordinate_present": has_depth,
                "selected_depth_m": float(dataset.depth.values[0])
                if has_depth and dataset.depth.size == 1
                else None,
                "depth_semantics": "2D horizontal surface velocity"
                if kind == "currents" and not has_depth
                else None,
                "time_coverage_valid": bool(time_ready),
                "geographic_coverage_valid": bool(area_ready),
                "patch_coverage_valid": bool(patch_area_ready),
                "units_valid": bool(units_ready),
                "time_axis_valid": bool(time_axis_ready),
            }
            if not time_ready:
                result[kind]["reason"] = (
                    "source timestamps do not cover July 2019 incident interval"
                )
            elif not area_ready:
                result[kind]["reason"] = (
                    "grid centers cover the patch but not the full requested bbox"
                )
    return result


def coastline_verify(root: Path) -> dict[str, Any]:
    from shapely.geometry import box, shape

    files = source_files(root)["coastline"]
    if len(files) != 1:
        return {"status": "MISSING"}
    payload = json.loads(files[0].read_text())
    crs = payload.get("crs", {}).get("properties", {}).get("name")
    geometries = [shape(feature["geometry"]) for feature in payload.get("features", [])]
    valid = (
        crs == "EPSG:4326"
        and bool(geometries)
        and all(g.is_valid and g.geom_type == "LineString" for g in geometries)
    )
    overlap = valid and any(g.intersects(box(*ROI)) for g in geometries)
    return {
        "status": "READY" if valid and overlap else "VALID_OUTSIDE_ROI" if valid else "INVALID",
        "path": str(files[0]),
        "crs": crs,
        "feature_count": len(geometries),
        "valid_geometry": bool(valid),
        "roi_overlap": bool(overlap),
        "reason": None if overlap else "no coastline segment intersects requested incident ROI",
    }


def synthetic_ais(root: Path) -> dict[str, Any]:
    """Six bounded, constant-speed demonstration routes; seed is fixed and recorded."""
    target = root / "raw/ais/synthetic_ais_20190702_20190705.csv"
    metadata = target.with_name("synthetic_ais_metadata.json")
    if target.exists():
        if not metadata.exists() or json.loads(metadata.read_text()).get("synthetic") is not True:
            raise ValueError("existing AIS CSV lacks synthetic provenance")
        return cast(dict[str, Any], json.loads(metadata.read_text()))
    rng = np.random.default_rng(42)
    routes = [
        ("900000101", "Demo tanker A", "tanker", (34.01, 32.83), (34.27, 33.13), 12.0, 0),
        ("900000102", "Demo cargo B", "cargo", (33.82, 32.68), (34.10, 33.03), 13.0, 4),
        ("900000103", "Demo tanker C", "tanker", (34.18, 32.72), (34.37, 32.96), 11.0, 32),
        ("900000104", "Demo vessel D", "general cargo", (34.20, 33.11), (33.98, 32.84), 10.0, 2),
        ("900000105", "Demo vessel E", "cargo", (33.60, 33.34), (33.80, 33.48), 12.0, 12),
        ("900000106", "Demo tanker F", "tanker", (33.91, 32.70), (34.12, 33.01), 10.0, 8),
    ]
    rows = []
    anchor = datetime(2019, 7, 2, 15, 40, 12, tzinfo=UTC)
    for mmsi, name, vessel_type, origin, destination, nominal_knots, offset_hours in routes:
        # Routes progress at a constant bearing; no jumps are introduced across an AIS gap.
        longitude_delta = (destination[0] - origin[0]) * 92.5
        latitude_delta = (destination[1] - origin[1]) * 111.2
        distance_km = math.hypot(longitude_delta, latitude_delta)
        hours = distance_km / (nominal_knots * 1.852)
        bearing = math.degrees(math.atan2(longitude_delta, latitude_delta)) % 360
        step_count = max(2, math.ceil(hours * 4))
        for index in range(step_count + 1):
            if mmsi == "900000106" and step_count // 3 < index < 2 * step_count // 3:
                continue
            fraction = index / step_count
            stamp = anchor + timedelta(hours=offset_hours + fraction * hours)
            if stamp > datetime(2019, 7, 5, 17, 40, 12, tzinfo=UTC):
                continue
            rows.append(
                {
                    "vessel_id": mmsi,
                    "timestamp": stamp.isoformat().replace("+00:00", "Z"),
                    "longitude": f"{origin[0] + fraction * (destination[0] - origin[0]):.6f}",
                    "latitude": f"{origin[1] + fraction * (destination[1] - origin[1]):.6f}",
                    "speed_over_ground_knots": f"{nominal_knots + rng.normal(0, 0.15):.2f}",
                    "course_over_ground_degrees": f"{bearing:.2f}",
                    "heading_degrees": f"{bearing:.2f}",
                    "vessel_name": name,
                    "vessel_type": vessel_type,
                }
            )
    rows.sort(key=lambda row: (row["timestamp"], row["vessel_id"]))
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {
        "synthetic": True,
        "historical_observation": False,
        "purpose": "hackathon pipeline demonstration",
        "seed": 42,
        "display_label": "Synthetic AIS demonstration",
        "vessel_count": len(routes),
        "point_count": len(rows),
        "path": str(target),
        "status": "READY",
    }
    metadata.write_text(json.dumps(info, indent=2) + "\n")
    return info


def display_stretch(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    valid = values[np.isfinite(values)]
    low, high = np.percentile(valid, [2, 98])
    return cast(
        np.ndarray[Any, Any],
        np.clip((values - low) * 255 / max(high - low, 1e-6), 0, 255).astype("uint8"),
    )


def _calibration_lut(
    xml: bytes, rows: np.ndarray[Any, Any], cols: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from scipy.interpolate import RegularGridInterpolator

    root = ET.fromstring(xml)
    vectors = root.findall(".//calibrationVector")
    lines = np.array([float(vector.findtext("line", "nan")) for vector in vectors])
    pixel_vectors = [np.fromstring(vector.findtext("pixel", ""), sep=" ") for vector in vectors]
    values = [np.fromstring(vector.findtext("sigmaNought", ""), sep=" ") for vector in vectors]
    if not len(vectors) or not all(
        np.array_equal(pixel_vectors[0], pixels) for pixels in pixel_vectors
    ):
        raise ValueError("calibration vectors are not on one consistent range grid")
    if any(len(pixels) != len(value) for pixels, value in zip(pixel_vectors, values, strict=True)):
        raise ValueError("calibration vector length mismatch")
    interpolator = RegularGridInterpolator(
        (lines, pixel_vectors[0]), np.array(values), bounds_error=False, fill_value=None
    )
    result = interpolator(np.column_stack((rows.ravel(), cols.ravel()))).reshape(rows.shape)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("invalid sigmaNought calibration LUT")
    return cast(np.ndarray[Any, Any], result)


def sar_prepare(
    root: Path, *, product: str = PRODUCT, patch: tuple[tuple[float, float], ...] = PATCH
) -> dict[str, Any]:
    """Apply product sigma0 LUT then GCP-interpolate onto PANGAEA footprint.

    SNAP remains unavailable/unconfigured. This explicitly documented route is
    approximate relative to the unknown Zenodo training raster preparation.
    """
    import rasterio
    from PIL import Image
    from rasterio import Affine
    from rasterio.windows import Window
    from scipy.interpolate import LinearNDInterpolator
    from scipy.ndimage import map_coordinates

    verified = sentinel_verify(root, product=product)
    if verified["status"] != "READY":
        raise ValueError("Sentinel product metadata failed validation")
    output = root / "prepared/model_input_vv_vh.tif"
    report = root / "prepared/preparation.json"
    with Path(verified["archive"]).open("rb") as stream:
        archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if output.exists() and report.exists():
        saved_report = json.loads(report.read_text())
        if (
            "source_archive_sha256" in saved_report
            and saved_report["source_archive_sha256"] != archive_sha256
        ):
            raise ValueError("source archive changed; prepared artifact is stale")
        with rasterio.open(output) as saved:
            if (
                saved.count == 2
                and saved.crs
                and saved.width == saved.height == 640
                and saved.descriptions == ("VV", "VH")
                and np.isfinite(saved.read()).all()
            ):
                if "source_archive_sha256" not in saved_report:
                    saved_report["source_archive_sha256"] = archive_sha256
                    report.write_text(json.dumps(saved_report, indent=2) + "\n")
                return cast(dict[str, Any], saved_report)
        raise ValueError("existing model input is invalid; refusing overwrite")
    zip_path = Path(verified["archive"])
    ul, ur, _, bl = patch
    transform = Affine(
        (ur[0] - ul[0]) / 640,
        (bl[0] - ul[0]) / 640,
        ul[0],
        (ur[1] - ul[1]) / 640,
        (bl[1] - ul[1]) / 640,
        ul[1],
    )
    grid_col, grid_row = np.meshgrid(
        np.arange(640, dtype="float64") + 0.5, np.arange(640, dtype="float64") + 0.5
    )
    longitudes = transform.c + transform.a * grid_col + transform.b * grid_row
    latitudes = transform.f + transform.d * grid_col + transform.e * grid_row
    source_names = verified["measurement"]
    bands, statistics = [], {}
    with zipfile.ZipFile(zip_path) as archive:
        for pol in ("VV", "VH"):
            path = f"/vsizip/{zip_path.resolve()}/{source_names[pol][0]}"
            with rasterio.open(path) as source:
                gcps, gcp_crs = source.gcps
                if str(gcp_crs) != "EPSG:4326":
                    raise ValueError("measurement GCPs must be EPSG:4326")
                points = np.array([(point.x, point.y) for point in gcps])
                source_coordinates = np.array([(point.col, point.row) for point in gcps])
                inverse = LinearNDInterpolator(points, source_coordinates)
                mapped = inverse(np.column_stack((longitudes.ravel(), latitudes.ravel())))
                if not np.isfinite(mapped).all():
                    raise ValueError("PANGAEA footprint exceeds Sentinel GCP convex hull")
                cols = mapped[:, 0].reshape(640, 640)
                rows = mapped[:, 1].reshape(640, 640)
                left = max(0, int(np.floor(cols.min())) - 3)
                top = max(0, int(np.floor(rows.min())) - 3)
                right = min(source.width, int(np.ceil(cols.max())) + 4)
                bottom = min(source.height, int(np.ceil(rows.max())) + 4)
                window = Window(left, top, right - left, bottom - top)
                dn = source.read(1, window=window)
                sampled = map_coordinates(
                    dn.astype("float32"),
                    [rows - top, cols - left],
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
            calibration_name = verified["calibration"][pol][0]
            lut = _calibration_lut(archive.read(calibration_name), rows, cols)
            sigma0 = sampled.astype("float64") ** 2 / lut**2
            if (sigma0 <= 0).any() or not np.isfinite(sigma0).all():
                raise ValueError(f"{pol} calibrated sigma0 contains zero/nonfinite cells")
            db = (10 * np.log10(sigma0)).astype("float32")
            bands.append(db)
            statistics[pol] = {
                "min": float(db.min()),
                "max": float(db.max()),
                "mean": float(db.mean()),
                "std": float(db.std()),
                "units": "sigma0 dB",
            }
    data = np.stack(bands)
    temporary = output.with_suffix(".tmp.tif")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        width=640,
        height=640,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
    ) as destination:
        destination.write(data)
        for index, pol in enumerate(("VV", "VH"), 1):
            destination.set_band_description(index, pol)
        destination.update_tags(
            BAND_ORDER="VV,VH", UNIT="sigma0 dB", PREPROCESSING_MATCH="approximate"
        )
    temporary.replace(output)
    for pol, band in zip(("vv", "vh"), data, strict=True):
        Image.fromarray(display_stretch(band)).save(root / f"prepared/{pol}_preview.png")
    info = {
        "source_product": product,
        "source_archive": str(zip_path),
        "source_archive_sha256": archive_sha256,
        "processing_stages": [
            "Sentinel-1 GRD measurement DN",
            "sigma0 from per-polarization product calibration LUT (DN squared / LUT squared)",
            "10*log10(sigma0) in dB",
            "GCP-based interpolation to PANGAEA patch footprint",
            "bilinear measurement sampling",
        ],
        "not_performed": [
            "precise orbit application",
            "thermal noise removal",
            "speckle filtering",
            "terrain correction",
            "land masking",
        ],
        "calibration_formula_source": "https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Level-1-Detailed-Algorithm-Definition",
        "crs": "EPSG:4326",
        "transform": list(transform)[:6],
        "dimensions": [640, 640],
        "band_order": ["VV", "VH"],
        "units": "sigma0 dB",
        "statistics": statistics,
        "preprocessing_match": "approximate",
        "reference_pixel_correspondence": (
            "not_established: source patch construction/resampling unknown"
        ),
        "geometric_method": (
            "linear inverse of Sentinel measurement GCPs to PANGAEA corner-derived affine grid"
        ),
    }
    report.write_text(json.dumps(info, indent=2) + "\n")
    return info


def run_detection(
    root: Path,
    checkpoint: Path,
    expected_hash: str,
    *,
    case_id: str = CASE,
    acquisition_start: str = "2019-07-05T15:40:12Z",
    acquisition_end: str = "2019-07-05T15:41:52Z",
) -> dict[str, Any]:
    """Run the existing RasterioSARReader and DeepLabV3PlusAdapter once."""
    import rasterio
    import yaml
    from PIL import Image

    from oilspill.adapters.detection.dense import DeepLabV3PlusAdapter
    from oilspill.adapters.sar.config import RasterioSARReaderConfig
    from oilspill.adapters.sar.reader import RasterioSARReader
    from oilspill.config import ComponentConfig
    from oilspill.domain.common import ArtifactRef, TimeRange
    from oilspill.requests import SARReadRequest

    prepared = root / "prepared/model_input_vv_vh.tif"
    result_path = root / "outputs/detection/result.json"
    if not prepared.exists():
        raise ValueError("validated VV/VH model input is required")
    with prepared.open("rb") as stream:
        prepared_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if result_path.exists():
        saved_result = json.loads(result_path.read_text())
        if saved_result.get("checkpoint_sha256") != expected_hash:
            raise ValueError("frozen inference checkpoint identity differs")
        if not all(
            (root / "outputs/detection" / name).is_file()
            for name in ("probability.tif", "mask.tif", "overlay.png")
        ):
            raise ValueError("frozen inference outputs are incomplete")
        if (
            "model_input_sha256" in saved_result
            and saved_result["model_input_sha256"] != prepared_sha256
        ):
            raise ValueError("prepared input changed; frozen inference result is stale")
        if "model_input_sha256" not in saved_result:
            saved_result["model_input_sha256"] = prepared_sha256
            result_path.write_text(json.dumps(saved_result, indent=2) + "\n")
        return cast(dict[str, Any], saved_result)

    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
            raise ValueError("checkpoint SHA256 mismatch")
    with rasterio.open(prepared) as source:
        if source.count != 2 or source.descriptions != ("VV", "VH") or not source.crs:
            raise ValueError("VV/VH input band/grid contract failed")
        if not np.isfinite(source.read()).all() or (source.read_masks() == 0).any():
            raise ValueError("model input contains nodata/nonfinite values")
        bounds = list(source.bounds)
        dimensions = [source.width, source.height]
        input_crs = str(source.crs)
    acquisition = TimeRange(
        start=datetime.fromisoformat(acquisition_start.replace("Z", "+00:00")).astimezone(UTC),
        end=datetime.fromisoformat(acquisition_end.replace("Z", "+00:00")).astimezone(UTC),
    )
    reader = RasterioSARReader(
        RasterioSARReaderConfig.model_validate(
            {
                "platform": "Sentinel-1A",
                "product_type": "IW_GRDH_1SDV",
                "acquisition": acquisition,
                "bands": {
                    "vv_band": 1,
                    "vh_band": 2,
                    "vv_unit": "sigma0 dB",
                    "vh_unit": "sigma0 dB",
                },
                "processing_level": "GRD calibrated approximate demo",
            }
        )
    )
    artifact = ArtifactRef(
        uri=prepared.resolve().as_uri(),
        media_type="image/tiff",
        sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
        byte_size=prepared.stat().st_size,
    )
    scene = reader.read(SARReadRequest(source=artifact, scene_id=case_id))
    settings = yaml.safe_load(Path("configs/models/deeplabv3plus_final.yaml").read_text())
    settings["settings"]["checkpoint"] = str(checkpoint.resolve())
    settings["settings"]["checkpoint_sha256"] = expected_hash
    settings["settings"]["output_root"] = str(root / "outputs/detection/adapter")
    settings["settings"]["spatial_transform"]["input_height"] = 640
    settings["settings"]["spatial_transform"]["input_width"] = 640
    settings["settings"]["device"] = "cpu"
    adapter = DeepLabV3PlusAdapter(ComponentConfig.model_validate(settings))
    started = time.perf_counter()
    adapter.load()
    detection = adapter.predict(scene)[0]
    runtime = time.perf_counter() - started
    from oilspill.artifacts import local_path_from_artifact

    output_dir = root / "outputs/detection"
    probability_tif = output_dir / "probability.tif"
    mask_tif = output_dir / "mask.tif"
    shutil.copyfile(
        local_path_from_artifact(detection.probability_raster.artifact), probability_tif
    )
    shutil.copyfile(local_path_from_artifact(detection.mask.artifact), mask_tif)
    with rasterio.open(probability_tif) as source:
        model_support = source.read(1)
        if source.crs.to_string() != input_crs or list(source.bounds) != bounds:
            raise ValueError("detector output georeferencing changed")
    with rasterio.open(mask_tif) as source:
        mask = source.read(1)
        if source.crs.to_string() != input_crs or list(source.bounds) != bounds:
            raise ValueError("detector mask georeferencing changed")
    with rasterio.open(prepared) as source:
        vv, vh = source.read()
    vv_display, vh_display = display_stretch(vv), display_stretch(vh)
    Image.fromarray(vv_display).save(output_dir / "vv.png")
    Image.fromarray(vh_display).save(output_dir / "vh.png")
    Image.fromarray((np.clip(model_support, 0, 1) * 255).astype("uint8")).save(
        output_dir / "probability.png"
    )
    Image.fromarray((mask == 1).astype("uint8") * 255).save(output_dir / "mask.png")
    overlay = np.stack((vv_display,) * 3, axis=2)
    overlay[mask == 1, 0] = np.maximum(overlay[mask == 1, 0], 220)
    overlay[mask == 1, 1:] = (overlay[mask == 1, 1:] * 0.45).astype("uint8")
    Image.fromarray(overlay).save(output_dir / "overlay.png")
    count = int((mask == 1).sum())
    info = {
        "case_id": case_id,
        "model": "deeplabv3plus_scse_boundary",
        "encoder": "resnet34",
        "checkpoint_sha256": expected_hash,
        "model_input_sha256": prepared_sha256,
        "input_dimensions": dimensions,
        "input_crs": input_crs,
        "input_bounds": bounds,
        "band_order": ["VV", "VH"],
        "preprocessing_status": "approximate",
        "threshold": 0.5,
        "oil_pixel_count": count,
        "oil_fraction": count / mask.size,
        "maximum_model_support": float(model_support.max()),
        "mean_model_support": float(model_support.mean()),
        "calibrated_probability": False,
        "inference_runtime_seconds": runtime,
        "device": "cpu",
        "reference_metrics": (
            "skipped: exact pixel correspondence to publisher patch not established"
        ),
        "output_paths": {
            name: str(output_dir / name)
            for name in (
                "probability.tif",
                "mask.tif",
                "probability.png",
                "mask.png",
                "vv.png",
                "vh.png",
                "overlay.png",
            )
        },
    }
    result_path.write_text(json.dumps(info, indent=2) + "\n")
    (output_dir / "inference.log").write_text(
        "DeepLabV3PlusAdapter(training_checkpoint=true) CPU inference completed once. "
        "Raw sigmoid output is uncalibrated model support.\n"
    )
    return info


def export_ui(root: Path) -> dict[str, Any]:
    target = Path("ui/data/instances") / CASE
    target.mkdir(parents=True, exist_ok=True)
    source = root / "outputs/detection"
    names = ("vv.png", "vh.png", "probability.png", "mask.png", "overlay.png", "result.json")
    exported = []
    for name in names:
        if (source / name).is_file():
            destination = target / name
            if destination.exists() and destination.read_bytes() == (source / name).read_bytes():
                exported.append(name)
                continue
            shutil.copyfile(source / name, destination)
            exported.append(name)
    info = {
        "case_id": CASE,
        "execution": "precomputed",
        "ais": {
            "source": "synthetic",
            "historical": False,
            "display_label": "Synthetic AIS demonstration",
        },
        "presentation_artifacts": exported,
        "detection_available": "result.json" in exported,
    }
    (target / "incident-demo.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
