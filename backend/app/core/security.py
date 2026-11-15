# hashlib: for SHA-256 hashing of refresh tokens
import hashlib
# secrets: cryptographically secure random values
import secrets
import uuid
from datetime import datetime, timedelta, timezone

# PyJWT: creates and verifies login tokens
import jwt
# argon2: the password hashing algorithm (slow on purpose, so guessing passwords is expensive)
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.core.config import settings

# argon2id hasher with the library's recommended settings
ph = PasswordHasher()

# A fake hash we check against when the email doesn't exist, so "unknown email"
# and "wrong password" take about the same time and can't be told apart by timing.
_DUMMY_HASH = ph.hash("timing-equalizer-not-a-real-password")


def hash_password(password: str) -> str:
    # Turns a password into a salted argon2 hash. Only the hash is ever stored.
    return ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    # True if the password matches the stored hash, False otherwise
    try:
        return ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def burn_dummy_verify(password: str) -> None:
    # Spends the same time as a real check (used when the email is unknown)
    verify_password(_DUMMY_HASH, password)


def needs_rehash(password_hash: str) -> bool:
    # True if the hash was made with older, weaker settings and should be upgraded at next login
    return ph.check_needs_rehash(password_hash)


def create_access_token(user_id: uuid.UUID) -> str:
    # Short-lived login token. It carries only the user id, NOT the role:
    # roles are read from the database on every request, so a demotion takes effect immediately.
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),          # who the token is for
        "type": "access",             # stops other token types being used as access tokens
        "jti": uuid.uuid4().hex,      # unique id for this token
        "iat": now,                   # issued at
        "exp": now + timedelta(minutes=settings.access_token_minutes),  # expires at
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    # Verifies signature and expiry, and returns the payload. Raises an error if anything is wrong.
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],  # pinned: blocks algorithm-confusion attacks
        options={"require": ["exp", "sub", "iat"]},  # these fields must be present
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("wrong token type")
    return payload


def new_refresh_token() -> str:
    # A random, unguessable string (384 bits). This goes into the user's cookie.
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw: str) -> str:
    # We store only this hash in the database. SHA-256 is fine here (not argon2)
    # because the token is long and random, unlike a human-chosen password.
    return hashlib.sha256(raw.encode()).hexdigest()