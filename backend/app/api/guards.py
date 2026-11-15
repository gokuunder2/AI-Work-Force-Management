from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

# get_current_user = check 1 (who are you?). Guards build on it.
from app.api.deps import get_current_user
from app.core.permissions import ROLE_PERMISSIONS, Permission
from app.db.session import get_db
from app.models.user import User
from app.services.audit import log_event


def _guard(perms: tuple[Permission, ...], require_all: bool):
    # Builds a FastAPI dependency that checks the given permissions
    def dependency(
        request: Request,
        user: User = Depends(get_current_user),   # runs check 1 first
        db: Session = Depends(get_db),
    ) -> User:
        # Look up what this user's CURRENT role allows (the role comes from the database)
        granted = ROLE_PERMISSIONS.get(user.role, frozenset())
        # all(): every permission needed. any(): at least one is enough.
        ok = all(p in granted for p in perms) if require_all else any(p in granted for p in perms)
        if not ok:
            # Record the denied attempt, so probing is visible in the audit log
            log_event(db, request, "permission_denied", actor_user_id=user.id, success=False,
                      details={"needed": [p.value for p in perms], "path": request.url.path})
            db.commit()
            # 403 = "I know who you are, but you may not do this"
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Forbidden")
        # Return the user so the route can use it
        return user

    return dependency


def require_permission(*perms: Permission):
    """The user must have ALL of these permissions."""
    return _guard(perms, require_all=True)


def require_any_permission(*perms: Permission):
    """The user must have AT LEAST ONE of these permissions."""
    return _guard(perms, require_all=False)