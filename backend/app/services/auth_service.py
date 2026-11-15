import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Request
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    burn_dummy_verify,
    hash_password,
    hash_refresh_token,
    needs_rehash,
    new_refresh_token,
    verify_password,
)
from app.models.auth import RefreshToken
from app.models.user import User
from app.services.audit import client_ip, client_ua, log_event


def _now() -> datetime:
    return datetime.now(timezone.utc)


def authenticate(db: Session, email: str, password: str, request: Request) -> User | None:
    """Returns the user on success, None on ANY failure. The caller must give the
    same generic error for every failure so attackers can't learn which emails exist.
    The real reason goes to the audit log only."""
    now = _now()
    # Find the user ignoring letter case. FOR UPDATE locks the row so two simultaneous
    # logins for one account can't both bypass the failure counter.
    user = db.scalar(
        select(User).where(func.lower(User.email) == email.lower()).with_for_update()
    )

    # Unknown email: do a fake password check so the response time looks the same
    if user is None:
        burn_dummy_verify(password)
        log_event(db, request, "login_failed", success=False, details={"reason": "unknown_user"})
        db.commit()
        return None

    # Account is currently locked: refuse without checking the password
    if user.locked_until and user.locked_until > now:
        log_event(db, request, "login_blocked", actor_user_id=user.id, success=False,
                  details={"reason": "locked"})
        db.commit()
        return None

    # Wrong password: count it, and lock the account when the limit is reached
    if not verify_password(user.password_hash, password):
        user.failed_login_attempts += 1
        log_event(db, request, "login_failed", actor_user_id=user.id, success=False,
                  details={"reason": "bad_password"})
        if user.failed_login_attempts >= settings.max_failed_logins:
            user.locked_until = now + timedelta(minutes=settings.lockout_minutes)
            user.failed_login_attempts = 0
            log_event(db, request, "account_locked", actor_user_id=user.id, success=False,
                      target_type="user", target_id=str(user.id),
                      details={"minutes": settings.lockout_minutes})
        db.commit()
        return None

    # Correct password but the account was disabled by an admin
    if not user.is_active:
        log_event(db, request, "login_failed", actor_user_id=user.id, success=False,
                  details={"reason": "inactive"})
        db.commit()
        return None

    # Success: reset the counters and record the login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    if needs_rehash(user.password_hash):  # quietly upgrade old hash settings
        user.password_hash = hash_password(password)
    log_event(db, request, "login_success", actor_user_id=user.id)
    db.commit()
    return user


def _issue_refresh_token(
    db: Session, user_id: uuid.UUID, family_id: uuid.UUID, request: Request
) -> tuple[str, RefreshToken]:
    # Creates a new refresh token. The raw value goes to the user's cookie;
    # only its hash is stored in the database.
    raw = new_refresh_token()
    row = RefreshToken(
        user_id=user_id,
        family_id=family_id,
        token_hash=hash_refresh_token(raw),
        expires_at=_now() + timedelta(days=settings.refresh_token_days),
        user_agent=client_ua(request),
        ip_address=client_ip(request),
    )
    db.add(row)
    db.flush()  # assigns row.id without committing yet
    return raw, row


def start_session(db: Session, user: User, request: Request) -> str:
    # A new login starts a new token family
    raw, _ = _issue_refresh_token(db, user.id, uuid.uuid4(), request)
    db.commit()
    return raw


def _revoke_family(db: Session, family_id: uuid.UUID, now: datetime) -> None:
    # Invalidates every still-valid token from one login
    db.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
    )


def rotate_refresh_token(db: Session, raw: str, request: Request) -> tuple[User, str] | None:
    # Swaps a valid refresh token for a new one. Each token works only once.
    now = _now()
    row = db.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(raw))
        .with_for_update()
    )
    if row is None:
        return None

    # An already-used token came back: either a thief or a replay.
    # Kill the entire chain so everyone must log in again.
    if row.revoked_at is not None:
        _revoke_family(db, row.family_id, now)
        log_event(db, request, "refresh_reuse_detected", actor_user_id=row.user_id,
                  success=False, details={"family_id": str(row.family_id)})
        db.commit()
        return None

    if row.expires_at <= now:
        return None

    # The user may have been disabled or locked since this session began
    user = db.get(User, row.user_id)
    if user is None or not user.is_active or (user.locked_until and user.locked_until > now):
        _revoke_family(db, row.family_id, now)
        db.commit()
        return None

    # Issue the replacement and mark the old token as used
    new_raw, new_row = _issue_refresh_token(db, user.id, row.family_id, request)
    row.revoked_at = now
    row.replaced_by_id = new_row.id
    db.commit()
    return user, new_raw


def revoke_session(db: Session, raw: str, request: Request) -> None:
    # Logout: kill the whole token family
    row = db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw))
    )
    if row is None:
        return
    _revoke_family(db, row.family_id, _now())
    log_event(db, request, "logout", actor_user_id=row.user_id)
    db.commit()