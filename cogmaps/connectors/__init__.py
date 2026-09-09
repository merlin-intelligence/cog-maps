"""Shared helpers for remote-source connectors (Google Drive, SharePoint)."""
from __future__ import annotations


def sanitize_remote_name(name: str) -> str:
    """Reduce a remote item's name (file or folder) to a single safe path component.

    Remote APIs return names verbatim from a third-party service (Drive,
    SharePoint) — a name of ``".."`` or containing ``/``/``\\`` would otherwise
    let ``os.path.join`` escape the intended download directory.
    """
    name = "".join(c if c not in '<>:"/\\|?*' else "_" for c in (name or "")).strip()
    if not name or set(name) <= {"."}:
        name = "_"
    return name
