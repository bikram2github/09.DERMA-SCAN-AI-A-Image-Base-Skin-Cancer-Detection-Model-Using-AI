import hashlib
import re
import secrets
import sqlite3
from typing import Optional

# Path to sqlite DB (kept compatible with existing file)
DB_PATH = "dermascan_users.db"


def _migrate_mobile_to_nullable(conn: sqlite3.Connection) -> None:
    """Ensure legacy DBs have mobile as nullable.

    Existing schema may have mobile defined as UNIQUE but NOT NULL. New UI stores
    mobile as NULL.
    """
    c = conn.cursor()
    try:
        c.execute("PRAGMA table_info(users)")
        cols = c.fetchall()
        # cols: (cid, name, type, notnull, dflt_value, pk)
        mobile_info = next((col for col in cols if col[1] == "mobile"), None)
        if mobile_info is None:
            return
        notnull = mobile_info[3]
        if notnull == 0:
            return  # already nullable

        # Rebuild table to make mobile nullable
        c.execute("ALTER TABLE users RENAME TO users_old")
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                mobile        TEXT    UNIQUE,
                password_hash TEXT    NOT NULL,
                email_verified INTEGER NOT NULL DEFAULT 1,
                verification_code TEXT,
                verification_expires_at INTEGER,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        c.execute(
            """
            INSERT INTO users (
                id, name, email, mobile, password_hash, email_verified,
                verification_code, verification_expires_at, created_at
            )
            SELECT
                id, name, email, NULL AS mobile, password_hash,
                1 AS email_verified,
                NULL AS verification_code,
                NULL AS verification_expires_at,
                created_at
            FROM users_old
            """
        )
        c.execute("DROP TABLE users_old")
    except sqlite3.Error:
        # Best-effort migration; app should remain usable.
        return


def _migrate_add_email_verification(conn: sqlite3.Connection) -> None:
    """Add email verification columns for legacy DBs (if missing)."""
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    cols = {col[1] for col in c.fetchall()}
    required = {"email_verified", "verification_code", "verification_expires_at"}
    if required.issubset(cols):
        return

    try:
        c.execute("ALTER TABLE users RENAME TO users_old")
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    UNIQUE NOT NULL,
                mobile        TEXT    UNIQUE,
                password_hash TEXT    NOT NULL,
                email_verified INTEGER NOT NULL DEFAULT 1,
                verification_code TEXT,
                verification_expires_at INTEGER,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        c.execute(
            """
            INSERT INTO users (
                id, name, email, mobile, password_hash, email_verified,
                verification_code, verification_expires_at, created_at
            )
            SELECT
                id, name, email, mobile, password_hash,
                1 AS email_verified,
                NULL AS verification_code,
                NULL AS verification_expires_at,
                created_at
            FROM users_old
            """
        )
        c.execute("DROP TABLE users_old")
    except sqlite3.Error:
        return


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            mobile        TEXT    UNIQUE,
            password_hash TEXT    NOT NULL,
            email_verified INTEGER NOT NULL DEFAULT 1,
            verification_code TEXT,
            verification_expires_at INTEGER,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    _migrate_mobile_to_nullable(conn)
    _migrate_add_email_verification(conn)
    conn.commit()
    conn.close()


# ── Password helpers ───────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}:{digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest = stored_hash.split(":", 1)
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == digest
    except Exception:
        return False


# ── Validation helpers ─────────────────────────────────────────────────────────
def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


# NOTE: mobile validation kept for legacy users; UI no longer uses it.
def validate_mobile(mobile: str) -> bool:
    """Accepts 10-digit mobile numbers (digits only)."""
    return bool(re.match(r"^\d{10}$", mobile.strip()))


# ── User operations ────────────────────────────────────────────────────────────
def create_user(name: str, email: str, password: str):
    """Register a new user (email + password).

    Email verification flow is currently disabled; accounts are active immediately.
    Returns (True, msg) or (False, err).
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute(
            """
            INSERT INTO users (name, email, mobile, password_hash, email_verified)
            VALUES (?, ?, NULL, ?, 1)
            """,
            (
                name.strip(),
                email.strip().lower(),
                hash_password(password),
            ),
        )
        conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as exc:
        msg = str(exc).lower()
        if "email" in msg:
            return False, "An account with this email address already exists."
        return False, "Registration failed. Please try again."
    finally:
        conn.close()


def resend_verification_code(email: str):
    """Deprecated: email verification removed."""
    return False, "Email verification has been disabled."


def verify_email(email: str, code: str):
    """Deprecated: email verification removed."""
    return False, "Email verification has been disabled."


def authenticate_user(email: str, password: str):
    """Login with email + password.

    Returns (True, user_dict) or (False, error message).
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, name, email, mobile, password_hash FROM users WHERE email = ?",
        (email.strip().lower(),),
    )
    row = c.fetchone()
    conn.close()

    if row is None:
        return False, "No account found with that email address."

    uid, name, email_db, mobile, stored_hash = row
    if not verify_password(password, stored_hash):
        return False, "Incorrect password. Please try again."

    return True, {"id": uid, "name": name, "email": email_db, "mobile": mobile}

