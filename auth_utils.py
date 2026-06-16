import os
import streamlit as st
import firebase_admin
from firebase_admin import auth, credentials


def init_firebase_admin() -> None:
    """Initialize Firebase Admin SDK.

    Expected env vars (recommended):
      - GOOGLE_APPLICATION_CREDENTIALS: path to service account JSON
    """
    if firebase_admin._apps:
        return

    # Use standard env var path for service account json
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa_path:
        raise RuntimeError(
            "Missing GOOGLE_APPLICATION_CREDENTIALS. "
            "Set it to the path of your Firebase service account JSON."
        )

    cred = credentials.Certificate(sa_path)
    firebase_admin.initialize_app(cred)


@st.cache_resource
def get_firebase_admin_initialized() -> bool:
    """Cache wrapper around init to avoid re-initialization."""
    init_firebase_admin()
    return True


def sign_out() -> None:
    """Clear local session auth state."""
    for k in ["id_token", "user_record"]:
        if k in st.session_state:
            del st.session_state[k]


def get_current_user() -> dict | None:
    """Return currently logged-in Firebase user (if any)."""
    token = st.session_state.get("id_token")
    if not token:
        return None

    get_firebase_admin_initialized()

    try:
        decoded = auth.verify_id_token(token)
        # Fetch the user record to ensure email verification state, uid, etc.
        user = auth.get_user(decoded["uid"])
        return {
            "uid": user.uid,
            "email": user.email,
            "email_verified": bool(user.email_verified),
            "name": user.display_name,
        }
    except Exception:
        # Token invalid or expired
        sign_out()
        return None


def require_verified_user() -> bool:
    user = get_current_user()
    return bool(user and user.get("email_verified"))

