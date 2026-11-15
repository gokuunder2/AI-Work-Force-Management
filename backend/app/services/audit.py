import ipaddress
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.models.auth import AuditLog


def client_ip(request: Request) -> str | None:
    # The caller's IP address, or None if it isn't a valid IP
    host = request.client.host if request.client else None
    if not host:
        return None
    try:
        return str(ipaddress.ip_address(host))  # the INET column needs a valid IP
    except ValueError:
        return None


def client_ua(request: Request) -> str | None:
    # Browser/device info, cut to 300 characters to match the column size
    ua = request.headers.get("user-agent")
    return ua[:300] if ua else None


def log_event(
    db: Session,
    request: Request,
    action: str,
    *,
    actor_user_id=None,
    success: bool = True,
    target_type: str | None = None,
    target_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Adds an audit row to the session. The caller commits, so the audit row and
    the change it describes succeed or fail together."""
    db.add(
        AuditLog(
            actor_user_id=actor_user_id,
            action=action,
            success=success,
            target_type=target_type,
            target_id=target_id,
            ip_address=client_ip(request),
            user_agent=client_ua(request),
            details=details,
        )
    )