"""Unit tests for cogmaps.ui.auth: hashing, fail-closed DB loading, rate limiting,
and the per-user collection ownership resolution.

These exercise the module's logic directly rather than the login form's rendering
(which needs a real Streamlit script run) — this logic (fail-open auth, weak
hashing, prefix-collision ownership) is security-critical.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import streamlit as st

from cogmaps.ui import auth


@pytest.fixture(autouse=True)
def _isolated_user_db(tmp_path, monkeypatch):
    """Point USER_DB_PATH at a throwaway file and reset session/rate-limit state."""
    db_path = tmp_path / "users.json"
    monkeypatch.setattr(auth, "USER_DB_PATH", str(db_path))
    monkeypatch.setattr(st, "session_state", {})
    auth._failed_logins.clear()
    yield db_path


# ── hash_password / verify_password ──────────────────────────────────────

def test_hash_password_is_salted():
    h1 = auth.hash_password("correct horse battery staple")
    h2 = auth.hash_password("correct horse battery staple")
    assert h1 != h2, "two hashes of the same password must differ (random salt)"
    assert h1.startswith("pbkdf2$")


def test_verify_password_accepts_correct_and_rejects_wrong():
    hashed = auth.hash_password("hunter2")
    assert auth.verify_password("hunter2", hashed)
    assert not auth.verify_password("wrong", hashed)


def test_verify_password_backward_compatible_with_legacy_sha256():
    legacy_hash = hashlib.sha256(b"oldscheme").hexdigest()
    assert auth.verify_password("oldscheme", legacy_hash)
    assert not auth.verify_password("wrong", legacy_hash)


def test_verify_password_rejects_malformed_pbkdf2_string():
    assert not auth.verify_password("anything", "pbkdf2$not$enough$fields$here")


# ── load_user_db / save_user_db ──────────────────────────────────────────

def test_load_user_db_bootstraps_empty_db_when_file_absent(tmp_path):
    db = auth.load_user_db()
    assert db == {"admins": [], "users": {}}
    assert (tmp_path / "users.json").exists()


def test_load_user_db_migrates_v1_flat_format():
    db_path = auth.USER_DB_PATH
    with open(db_path, "w") as f:
        json.dump({"alice": "somehash"}, f)
    db = auth.load_user_db()
    assert db == {"admins": [], "users": {"alice": "somehash"}}


def test_load_user_db_raises_on_corrupted_file_instead_of_returning_empty():
    """The critical fail-open bug: a corrupted DB must never be treated as
    'no users configured' (which would let everyone in unauthenticated)."""
    with open(auth.USER_DB_PATH, "w") as f:
        f.write("{not valid json at all")
    with pytest.raises(auth.UserDBError):
        auth.load_user_db()


def test_save_user_db_is_atomic_no_leftover_tmp_file():
    import os
    auth.save_user_db({"admins": [], "users": {"bob": "x"}})
    assert not os.path.exists(auth.USER_DB_PATH + ".tmp")
    with open(auth.USER_DB_PATH) as f:
        assert json.load(f) == {"admins": [], "users": {"bob": "x"}}


# ── is_admin fails closed on a corrupted DB ──────────────────────────────

def test_is_admin_false_when_db_corrupted():
    st.session_state["authenticated_user"] = "alice"
    with open(auth.USER_DB_PATH, "w") as f:
        f.write("not json")
    assert auth.is_admin() is False


def test_is_admin_true_for_listed_admin():
    auth.save_user_db({"admins": ["alice"], "users": {"alice": "x", "bob": "y"}})
    st.session_state["authenticated_user"] = "alice"
    assert auth.is_admin() is True
    st.session_state["authenticated_user"] = "bob"
    assert auth.is_admin() is False


# ── rate limiting ─────────────────────────────────────────────────────────

def test_rate_limiting_locks_out_after_max_attempts():
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        assert auth._is_rate_limited("alice") == 0.0
        auth._record_failed_login("alice")
    assert auth._is_rate_limited("alice") > 0.0


def test_rate_limiting_is_per_user():
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        auth._record_failed_login("alice")
    assert auth._is_rate_limited("alice") > 0.0
    assert auth._is_rate_limited("bob") == 0.0


def test_clear_failed_logins_resets_lockout():
    for _ in range(auth._MAX_LOGIN_ATTEMPTS):
        auth._record_failed_login("alice")
    auth._clear_failed_logins("alice")
    assert auth._is_rate_limited("alice") == 0.0


# ── collection ownership: bob / bob_admin prefix collision ──────────────

def test_owning_user_resolves_prefix_collision_to_longest_match():
    usernames = ["bob", "bob_admin", "alice"]
    assert auth._owning_user("bob_admin_secrets", usernames) == "bob_admin"
    assert auth._owning_user("bob_reports", usernames) == "bob"
    assert auth._owning_user("alice_notes", usernames) == "alice"
    assert auth._owning_user("nobody_x", usernames) is None


def test_owns_collection_bob_cannot_claim_bob_admins_collection():
    auth.save_user_db({"admins": [], "users": {"bob": "x", "bob_admin": "y"}})
    st.session_state["authenticated_user"] = "bob"
    assert auth.owns_collection("bob_reports") is True
    assert auth.owns_collection("bob_admin_secrets") is False


def test_list_visible_collections_excludes_other_users_and_includes_public():
    auth.save_user_db({"admins": [], "users": {"bob": "x", "bob_admin": "y"}})
    st.session_state["authenticated_user"] = "bob"
    all_qdrant = ["bob_reports", "bob_admin_secrets", "public_shared", "alice_private"]
    visible = auth.list_visible_collections(all_qdrant)
    assert ("reports", "bob_reports") in visible
    assert ("[public] shared", "public_shared") in visible
    assert not any(q == "bob_admin_secrets" for _, q in visible)
    assert not any(q == "alice_private" for _, q in visible)


def test_guest_mode_no_users_configured_can_see_own_collections():
    """With no users configured (open-access mode), current_user() returns
    "guest", which is never a registered user. _owning_user must still
    resolve "guest_..." collections to "guest", or a user's own collections
    become invisible to them right after ingesting."""
    auth.save_user_db({"admins": [], "users": {}})
    # No "authenticated_user" set in session_state -> current_user() falls
    # back to "guest", exactly like the real no-auth code path.
    all_qdrant = ["guest_my_collection", "public_shared"]
    visible = auth.list_visible_collections(all_qdrant)
    assert ("my_collection", "guest_my_collection") in visible
    assert auth.owns_collection("guest_my_collection") is True
    assert auth.display_name_from("guest_my_collection") == "my_collection"


def test_display_name_from_returns_none_for_foreign_collection():
    auth.save_user_db({"admins": [], "users": {"bob": "x", "bob_admin": "y"}})
    st.session_state["authenticated_user"] = "bob"
    assert auth.display_name_from("bob_reports") == "reports"
    assert auth.display_name_from("bob_admin_secrets") is None


# ── logout ────────────────────────────────────────────────────────────────

def test_logout_clears_session_state():
    st.session_state["password_correct"] = True
    st.session_state["authenticated_user"] = "bob"
    st.session_state["last_activity"] = 123.0
    st.session_state["gdrive_client"] = object()
    auth.logout()
    assert "password_correct" not in st.session_state
    assert "authenticated_user" not in st.session_state
    assert "last_activity" not in st.session_state
    assert "gdrive_client" not in st.session_state
