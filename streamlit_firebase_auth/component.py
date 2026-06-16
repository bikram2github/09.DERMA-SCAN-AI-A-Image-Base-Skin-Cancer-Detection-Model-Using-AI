import json
from pathlib import Path

import streamlit as st
import streamlit_component_lib


_RELEASE_DIR = Path(__file__).parent / "frontend" / "build"


# `declare_component` expects a folder that contains an `index.html` (and other static assets).
firebase_auth_component = streamlit_component_lib.declare_component(
    "firebase_auth_component",
    path=str(_RELEASE_DIR),
)


def get_firebase_auth_state(
    *,
    api_key: str,
    auth_domain: str,
    project_id: str,
    app_id: str,
):
    """
    Renders the embedded Firebase email/password auth UI.
    Returns dict or None.

    Output (dict):
      {
        "idToken": "...",
        "emailVerified": true/false,
        "email": "user@example.com",
        "status": "logged_in" | "logged_out" | "error" | "needs_verify"
      }
    """
    payload = firebase_auth_component(
        apiKey=api_key,
        authDomain=auth_domain,
        projectId=project_id,
        appId=app_id,
        default=None,
    )
    if payload is None:
        return None
    if isinstance(payload, str):
        # streamlit-component-lib may sometimes send strings
        try:
            return json.loads(payload)
        except Exception:
            return {"status": "error", "message": payload}
    return payload
