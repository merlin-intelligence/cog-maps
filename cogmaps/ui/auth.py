"""Multi-user authentication backed by a JSON DB on disk."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time

import streamlit as st

USER_DB_PATH = "user_data/users.json"
PUBLIC_PREFIX = "public_"

_PBKDF2_ITERATIONS = 260_000

# In-memory login rate limiting (per process — resets on restart, which is
# acceptable since it only needs to slow down online brute-forcing, not
# survive a redeploy). Keyed by the *attempted* username.
_MAX_LOGIN_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300
_failed_logins: dict[str, list[float]] = {}

# Sessions with no activity for this long are logged out on their next page load.
_SESSION_TIMEOUT_SECONDS = 8 * 3600


class UserDBError(RuntimeError):
    """Raised when the user DB file exists but cannot be read/parsed.

    Callers must fail closed (deny access) on this — never treat a corrupted
    or unreadable DB as "no users configured, so let everyone in".
    """


def hash_password(password: str) -> str:
    """Salted PBKDF2-HMAC-SHA256 hash, stored as ``pbkdf2$<iterations>$<salt>$<hash>``."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"pbkdf2${_PBKDF2_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check ``password`` against a stored hash, salted (current) or legacy plain SHA-256."""
    if stored.startswith("pbkdf2$"):
        try:
            _, iterations_s, salt, expected_hex = stored.split("$")
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(salt), int(iterations_s)
            )
            return hmac.compare_digest(dk.hex(), expected_hex)
        except (ValueError, TypeError):
            return False
    # Unsalted SHA-256 hash format, still accepted for backward compatibility.
    legacy_hex = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy_hex, stored)


def _is_rate_limited(user: str) -> float:
    """Return remaining lockout seconds for ``user`` (0 if not locked out)."""
    now = time.monotonic()
    attempts = [t for t in _failed_logins.get(user, []) if now - t < _LOGIN_WINDOW_SECONDS]
    _failed_logins[user] = attempts
    if len(attempts) < _MAX_LOGIN_ATTEMPTS:
        return 0.0
    return max(0.0, _LOGIN_WINDOW_SECONDS - (now - attempts[0]))


def _record_failed_login(user: str) -> None:
    _failed_logins.setdefault(user, []).append(time.monotonic())


def _clear_failed_logins(user: str) -> None:
    _failed_logins.pop(user, None)


def _is_v2(db: dict) -> bool:
    return "users" in db and isinstance(db.get("users"), dict)


def load_user_db() -> dict:
    """Load the user DB. Returns v2 format: {"admins": [...], "users": {username: hash}}.

    Raises ``UserDBError`` if the file exists but is unreadable or corrupted.
    """
    if not os.path.exists(USER_DB_PATH):
        os.makedirs("user_data", exist_ok=True)
        initial: dict = {"admins": [], "users": {}}
        try:
            if "USERS" in st.secrets:
                initial["users"] = {u: hash_password(p) for u, p in st.secrets["USERS"].items()}
            elif "APP_PASSWORD" in st.secrets:
                initial["users"] = {
                    st.secrets["APP_USERNAME"].strip():
                        hash_password(st.secrets["APP_PASSWORD"].strip())
                }
        except Exception:
            pass
        save_user_db(initial)
        return initial
    try:
        with open(USER_DB_PATH, "r") as f:
            db = json.load(f)
    except Exception as e:
        raise UserDBError(f"could not read {USER_DB_PATH}: {e}") from e
    if _is_v2(db):
        return db
    # Migrate v1 flat {user: hash} to v2
    return {"admins": [], "users": db}


def save_user_db(db: dict) -> None:
    """Write the user DB atomically (temp file + rename) to avoid partial writes."""
    tmp_path = f"{USER_DB_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp_path, USER_DB_PATH)


def logout() -> None:
    """Clear the authenticated session, forcing a fresh login on next render."""
    st.session_state.pop("password_correct", None)
    st.session_state.pop("authenticated_user", None)
    st.session_state.pop("last_activity", None)
    st.session_state.pop("gdrive_client", None)


def check_password() -> bool:
    """Render the login UI and return True iff the user is authenticated."""
    try:
        db = load_user_db()
    except UserDBError as e:
        st.error(f"🔒 Authentication unavailable: {e}. Contact an administrator.")
        return False
    users = db["users"]
    if not users:
        return True

    if not st.session_state.get("password_correct", False):
        _, mid, _ = st.columns([1, 1.2, 1])
        with mid:
            st.markdown("""
                <div style="text-align: center; margin-top: 5rem; margin-bottom: 2rem;">
                    <h1 style="font-family:'Playfair Display',serif; font-weight:900;
                               font-size:3rem; color:#2a1f18; margin-bottom:0;">eigenmind</h1>
                    <div style="font-family:'DM Mono',monospace; letter-spacing:0.2em;
                                color:#c44a28; font-size:0.8rem; text-transform:uppercase;">
                        accelerate clarity
                    </div>
                </div>
            """, unsafe_allow_html=True)
            with st.container(border=True):
                with st.form("login_form", clear_on_submit=False):
                    user = st.text_input("identity", key="login_username", placeholder="username")
                    pwd = st.text_input("credential", type="password",
                                        key="login_password", placeholder="password")
                    submitted = st.form_submit_button("sign in", use_container_width=True)
                if submitted:
                    user = (user or "").strip()
                    pwd = (pwd or "").strip()
                    lockout_remaining = _is_rate_limited(user)
                    if lockout_remaining > 0:
                        st.error(
                            f"🔒 Too many failed attempts. Try again in "
                            f"{int(lockout_remaining) // 60 + 1} min."
                        )
                    elif user in users and verify_password(pwd, users[user]):
                        _clear_failed_logins(user)
                        # Transparently upgrade legacy unsalted hashes on successful login.
                        if not users[user].startswith("pbkdf2$"):
                            db["users"][user] = hash_password(pwd)
                            save_user_db(db)
                        st.session_state["password_correct"] = True
                        st.session_state["authenticated_user"] = user
                        st.session_state.pop("login_password", None)
                        st.session_state.pop("login_username", None)
                        st.rerun()
                    else:
                        _record_failed_login(user)
                        st.session_state["password_correct"] = False
                        st.error("😕 Authentication failed. Access denied.")
                st.markdown("""
                    <div style="margin-top: 2rem; font-family:'DM Mono',monospace;
                                font-size:0.6rem; color:#8a6a50; text-align:center; opacity:0.6;">
                        SECURE ACCESS POINT · PX-V EIGENMIND v2.1 (Multi-User)
                    </div>
                """, unsafe_allow_html=True)
        return False

    # Already authenticated for this session — enforce an inactivity timeout.
    now = time.time()
    last_seen = st.session_state.get("last_activity", now)
    if now - last_seen > _SESSION_TIMEOUT_SECONDS:
        logout()
        st.warning("🔒 Session expired after inactivity — please sign in again.")
        st.rerun()
    st.session_state["last_activity"] = now
    return True


def current_user() -> str:
    return st.session_state.get("authenticated_user", "guest")


def is_admin() -> bool:
    """Return True if the current user has admin privileges."""
    try:
        db = load_user_db()
    except UserDBError:
        return False
    return current_user() in db.get("admins", [])


def get_user_token_path() -> str:
    """Per-user file used to cache OAuth tokens."""
    user_dir = os.path.join("user_data", current_user())
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "gdrive_token.json")


def qdrant_collection_for(display_name: str) -> str:
    """Namespace a display name with the current user (private collection)."""
    return f"{current_user()}_{display_name}"


def public_qdrant_name_for(display_name: str) -> str:
    """Return the Qdrant collection name for a public collection."""
    return f"{PUBLIC_PREFIX}{display_name}"


def _all_usernames() -> list[str]:
    """Registered usernames, plus the current session's user.

    The current user is always included even when they're not in
    ``users.json`` — e.g. the "no users configured" open-access mode, where
    ``current_user()`` returns ``"guest"`` but ``"guest"`` is never a
    registered user. This ensures ``_owning_user`` can match that user's own
    collections (candidates starting with ``"guest_"``), keeping their
    private collections visible to them.
    """
    try:
        db = load_user_db()
    except UserDBError:
        usernames = []
    else:
        usernames = list(db.get("users", {}).keys())
    current = current_user()
    if current not in usernames:
        usernames.append(current)
    return usernames


def _owning_user(qdrant_name: str, usernames: list[str]) -> str | None:
    """Resolve which user owns a private collection.

    Plain ``startswith(f"{user}_")`` is ambiguous when one username is itself
    a prefix of another (e.g. users "bob" and "bob_admin": collection
    "bob_admin_secrets" starts with both "bob_" and "bob_admin_"). Picking the
    *longest* matching username resolves it unambiguously — "bob_admin_secrets"
    belongs to "bob_admin", never to "bob".
    """
    candidates = [u for u in usernames if u and qdrant_name.startswith(f"{u}_")]
    return max(candidates, key=len) if candidates else None


def owns_collection(qdrant_name: str) -> bool:
    """Return True if the current user owns this private (non-public) collection."""
    return _owning_user(qdrant_name, _all_usernames()) == current_user()


def display_name_from(qdrant_name: str) -> str | None:
    """Return the display name if this collection belongs to the current user, else None."""
    owner = _owning_user(qdrant_name, _all_usernames())
    if owner is None or owner != current_user():
        return None
    return qdrant_name[len(owner) + 1:]


def list_visible_collections(all_qdrant: list[str]) -> list[tuple[str, str]]:
    """Return (display_label, qdrant_name) for collections visible to the current user.

    Includes the user's own private collections and all public collections.
    Public collections are labeled with a '[public] ' prefix in the display label.
    """
    user = current_user()
    usernames = _all_usernames()
    result: list[tuple[str, str]] = []
    for name in all_qdrant:
        if name.startswith(PUBLIC_PREFIX):
            result.append((f"[public] {name[len(PUBLIC_PREFIX):]}", name))
            continue
        owner = _owning_user(name, usernames)
        if owner == user:
            result.append((name[len(owner) + 1:], name))
    return sorted(result)
