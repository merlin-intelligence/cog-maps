"""Unit tests for cogmaps.connectors: the shared remote-filename sanitizer (path
traversal guard for Google Drive / SharePoint item names) and the Google Drive
folder-ID validation (query-injection guard).

The gdrive-specific tests are skipped when the optional `[gdrive]` extra isn't
installed, since `cogmaps.connectors.gdrive` is not part of the base/dev install.
"""
from __future__ import annotations

import pytest

from cogmaps.connectors import sanitize_remote_name

# ── sanitize_remote_name: shared by gdrive.py and sharepoint.py ──────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("report.pdf", "report.pdf"),
        ("..", "_"),
        ("", "_"),
        ("a/b\\c", "a_b_c"),
        ('weird<>:"|?*name', "weird_______name"),
    ],
)
def test_sanitize_remote_name(raw, expected):
    assert sanitize_remote_name(raw) == expected


def test_sanitize_remote_name_never_produces_a_path_separator():
    for raw in ["../../../etc/passwd", "a/../../b", "..\\..\\windows"]:
        safe = sanitize_remote_name(raw)
        assert "/" not in safe
        assert "\\" not in safe


def test_sanitize_remote_name_pure_dots_cannot_traverse():
    # A bare ".." (or "...") must not survive as a literal path component that
    # os.path.join could interpret as "go up a directory".
    for raw in ["..", "...", "."]:
        assert sanitize_remote_name(raw) != raw or set(sanitize_remote_name(raw)) != set(raw)


# ── gdrive: folder-ID validation (guards the Drive query string) ────────

googleapiclient = pytest.importorskip("googleapiclient", reason="optional [gdrive] extra not installed")


def test_gdrive_rejects_folder_id_containing_a_quote():
    from cogmaps.connectors.gdrive import GDriveClient

    client = GDriveClient(service=None)
    with pytest.raises(ValueError):
        client.download_folder("x' or fullText contains 'password", "/tmp/dest")


def test_gdrive_accepts_well_formed_folder_id():
    from cogmaps.connectors.gdrive import _DRIVE_ID_RE

    assert _DRIVE_ID_RE.match("1a2B3-c_D")
    assert not _DRIVE_ID_RE.match("has a space")
    assert not _DRIVE_ID_RE.match("")
