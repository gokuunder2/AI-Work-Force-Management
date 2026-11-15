# uuid: used for the type of the random IDs
import uuid
# datetime: the Python type for date + time values (lockout time, last login)
from datetime import datetime

# Column types and rules used to define the users table.
# func lets us call database functions, such as lower()
from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, Integer, String, func
# PostgreSQL's native UUID column type
from sqlalchemy.dialects.postgresql import UUID
# Mapped/mapped_column: define typed columns
from sqlalchemy.orm import Mapped, mapped_column

# Base: parent of all tables; the two mixins add the id and created/updated columns
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
# The allowed roles (admin, hr, manager, employee, sales)
from app.models.enums import Role


# USERS TABLE: login accounts. Gets a random UUID id and created_at/updated_at automatically.
class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    # The real table name in PostgreSQL
    __tablename__ = "users"

    # Login email, required, up to 254 characters (the maximum length of a valid email).
    # Uniqueness is enforced by the case-insensitive index at the bottom of this file.
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    # Argon2 hash of the password, never the password itself. The 255 limit leaves room for the hash.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # The user's single role. Only the values in Role are accepted.
    role: Mapped[Role] = mapped_column(
        # native_enum=False stores plain text (easier to change later);
        # values_callable stores "admin" rather than the Python name "ADMIN"
        Enum(Role, native_enum=False, length=20,
             values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    # Links this login to an HR record. Used for "employees see only their own record"
    # and for working out a manager's team. Optional: Admin and Sales accounts may have none.
    employee_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        # ondelete="SET NULL": if the employee row is deleted, the login stays but the link is cleared
        ForeignKey("employees.id", ondelete="SET NULL"),
        # unique=True: one employee record can belong to only one login
        unique=True,
    )
    # False disables the account without deleting it (the user can no longer log in)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Lockout counter: wrong passwords in a row. Reset to 0 on a successful login.
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # If set to a future time, the account is locked until then. Empty means not locked.
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # When the user last logged in successfully (useful for audits and spotting unused accounts)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # When the password was last changed (lets us expire old sessions after a change later)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# Case-insensitive uniqueness: "A@x.com" and "a@x.com" are the same account.
# The index is built on lower(email), so two emails that differ only in capital
# letters count as duplicates and the database rejects the second one.
Index("uq_users_email_lower", func.lower(User.email), unique=True)