# uuid: used for the type of the random IDs
import uuid
# datetime: the Python type for date + time values
from datetime import datetime
# Any: lets the audit "details" field hold flexible JSON data
from typing import Any

# Column types and rules used to define the tables.
# func lets us call database functions, such as now()
from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Identity, String, func
# PostgreSQL-specific types: INET (IP address), JSONB (JSON data), UUID
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
# Mapped/mapped_column: define typed columns
from sqlalchemy.orm import Mapped, mapped_column

# Base: parent of all tables; UUIDPrimaryKeyMixin adds a random UUID id column
from app.db.base import Base, UUIDPrimaryKeyMixin


# REFRESH_TOKENS TABLE: one row per login session token. Gets a random UUID id automatically.
class RefreshToken(UUIDPrimaryKeyMixin, Base):
    """Only a SHA-256 hash of the token is stored, so a DB leak doesn't leak sessions.
    `family_id` groups a rotation chain: if an already-used token is replayed,
    the whole family is revoked (theft detection)."""

    # The real table name in PostgreSQL
    __tablename__ = "refresh_tokens"

    # Which user this session belongs to. Required.
    user_id: Mapped[uuid.UUID] = mapped_column(
        # ondelete="CASCADE": if the user is deleted, their sessions are deleted too.
        # index=True: makes "all sessions of this user" lookups fast.
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Groups all tokens from one login. Each refresh creates a new token in the same family.
    # If an old (already used) token shows up again, we revoke the whole family.
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True, nullable=False)
    # SHA-256 hash of the token (64 hex characters). The real token only exists in the user's cookie.
    # unique=True: no two rows can share a hash, which also makes lookups fast.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # When the token was created. The database fills it in automatically.
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # When the token stops working, even if never used. Required.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Set when the token is used up or the user logs out. Empty means still valid.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The id of the token that replaced this one when it was refreshed (shows the rotation chain)
    replaced_by_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    # Browser/device info, cut to 300 characters. Lets users see where they are logged in.
    user_agent: Mapped[str | None] = mapped_column(String(300))
    # IP address the session started from, stored in PostgreSQL's native IP type
    ip_address: Mapped[str | None] = mapped_column(INET)


# AUDIT_LOG TABLE: permanent record of important events (logins, role changes, exports)
class AuditLog(Base):
    """Append-only (enforced by a DB trigger in the second migration)."""

    # The real table name in PostgreSQL
    __tablename__ = "audit_log"

    # Auto-numbered id (1, 2, 3...). Numbers are fine here because the log is never exposed
    # through URLs, and they keep the events in order.
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # When the event happened. The database fills it in, so the app can't fake it.
    # index=True: makes "events in this time range" searches fast.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    # SET NULL so deleting a user never deletes their history.
    # Who did it. Empty for events with no known user (for example, a login with an unknown email).
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    # e.g. login_success, login_failed, account_locked, role_changed, data_export
    # What happened. Indexed so "show all failed logins" is fast.
    action: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    # What kind of thing was affected (for example "user") and which one. Both optional.
    target_type: Mapped[str | None] = mapped_column(String(50))
    target_id: Mapped[str | None] = mapped_column(String(64))
    # True if the action succeeded, False if it failed or was blocked
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # IP address and browser info of whoever caused the event
    ip_address: Mapped[str | None] = mapped_column(INET)
    user_agent: Mapped[str | None] = mapped_column(String(300))
    # Never put passwords/tokens/PII payloads here: IDs and old/new role values only.
    # Flexible extra info stored as JSON, such as {"old_role": "hr", "new_role": "manager"}
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)