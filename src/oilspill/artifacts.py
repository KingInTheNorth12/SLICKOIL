"""Infrastructure-neutral helpers for immutable artifact references."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse

from oilspill.domain.common import ArtifactRef


class UnsupportedArtifactURIError(ValueError):
    """An operation requiring a local file received another URI scheme."""


def local_path_from_artifact(artifact: ArtifactRef) -> Path:
    """Resolve a local path or ``file:`` URI without applying domain-specific semantics."""

    parsed = urlparse(artifact.uri)
    if parsed.scheme not in {"", "file"}:
        raise UnsupportedArtifactURIError(
            f"artifact requires a local file, got URI scheme {parsed.scheme!r}"
        )
    return Path(unquote(parsed.path if parsed.scheme == "file" else artifact.uri))


def write_artifact(path: Path, content: bytes, media_type: str = "application/json") -> ArtifactRef:
    """Append-only local artifact creation shared by generic ensemble services."""
    with path.open("xb") as stream:
        stream.write(content)
    return ArtifactRef(
        uri=path.resolve().as_uri(),
        sha256=sha256(content).hexdigest(),
        byte_size=len(content),
        media_type=media_type,
    )


def read_artifact(artifact: ArtifactRef) -> bytes:
    """Read a local artifact and verify its content identity without network access."""
    content = local_path_from_artifact(artifact).read_bytes()
    if sha256(content).hexdigest() != artifact.sha256:
        raise ValueError("artifact checksum mismatch")
    return content
