"""High-precision filename-extension rules for LionAI's hybrid classifier.

Only unambiguous, conventional extensions are included. Ambiguous suffixes such
as json, sql, html, ts, and gz intentionally remain model-only.
"""
from __future__ import annotations

from pathlib import Path

EXTENSION_RULES: dict[str, str] = {
    **{ext: "Images" for ext in (
        "jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff",
        "ico", "avif", "heic", "psd", "xcf", "exr", "eps",
    )},
    **{ext: "Audio" for ext in (
        "mp3", "wav", "flac", "m4a", "aac", "aiff", "aif", "opus",
    )},
    **{ext: "Video" for ext in (
        "mp4", "mkv", "mov", "avi", "webm", "m4v", "mpeg", "mpg",
    )},
    **{ext: "Archives" for ext in (
        "zip", "7z", "rar", "tar", "tgz", "bz2", "xz", "zst",
    )},
    **{ext: "Fonts" for ext in ("ttf", "otf", "woff", "woff2", "eot", "pfb")},
    **{ext: "Installers" for ext in (
        "exe", "msi", "msix", "dmg", "deb", "rpm", "apk", "aab",
        "appimage", "flatpak", "snap",
    )},
    **{ext: "Databases" for ext in ("sqlite", "sqlite3", "db", "duckdb", "mdb", "bson", "rdb")},
    **{ext: "Data" for ext in ("csv", "tsv", "parquet", "feather", "avro", "orc", "geojson", "ndjson")},
    **{ext: "Documents" for ext in ("doc", "docx", "odt", "rtf", "epub", "ppt", "pptx", "tex", "rst")},
    **{ext: "Source" for ext in (
        "py", "pyi", "jsx", "tsx", "java", "c", "h", "cc", "cpp", "hpp",
        "cs", "go", "rs", "rb", "php", "swift", "kt", "kts", "scala",
        "sh", "ps1", "bat", "lua", "dart", "vue", "svelte",
    )},
}


def extension_rule(filename: str) -> str | None:
    """Return a high-precision category for a plain filename, or None."""
    if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
        return None
    suffix = Path(filename).suffix.casefold().lstrip(".")
    return EXTENSION_RULES.get(suffix)
