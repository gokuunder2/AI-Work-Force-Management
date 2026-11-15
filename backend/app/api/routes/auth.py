from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.rate_limit import limiter
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse, UserOut
from app.services.auth_service import (
    authenticate,
    revoke_session,
    rotate_refresh_token,
    start_session,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

COOKIE_NAME = "refresh_token"
# The cookie is only sent to /api/auth/*, not to every API call
COOKIE_PATH = "/api/auth"


def _set_refresh_cookie(response: Response, raw: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=raw,
        max_age=settings.refresh_token_days * 86400,
        httponly=True,                   # JavaScript can't read it, so XSS can't steal it
        secure=settings.cookie_secure,   # HTTPS only (false for local development)
        samesite="strict",               # never sent on cross-site requests, which blocks CSRF
        path=COOKIE_PATH,
    )


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=settings.access_token_minutes * 60,
        user=UserOut.model_validate(user),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")  # max 10 login attempts per minute per IP
def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    db: Session = Depends(get_db),
):
    user = authenticate(db, body.email, body.password, request)
    if user is None:
        # One generic message for every failure (wrong password, unknown email, locked, inactive)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    _set_refresh_cookie(response, start_session(db, user, request))
    return _token_response(user)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("30/minute")
def refresh(
    request: Request,
    response: Response,
    refresh_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    result = rotate_refresh_token(db, refresh_token, request) if refresh_token else None
    if result is None:
        failed = JSONResponse({"detail": "Session expired"}, status_code=401)
        failed.delete_cookie(COOKIE_NAME, path=COOKIE_PATH)
        return failed
    user, new_raw = result
    _set_refresh_cookie(response, new_raw)
    return _token_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
def logout(
    request: Request,
    refresh_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
):
    if refresh_token:
        revoke_session(db, refresh_token, request)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(COOKIE_NAME, path=COOKIE_PATH)
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    # Returns the logged-in user. Needs a valid access token.
    return user