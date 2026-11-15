"""Create the first Admin. Run from backend/:  python -m scripts.create_admin"""
import getpass
import sys

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy import func, select

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models.auth import AuditLog
from app.models.enums import Role
from app.models.user import User

MIN_PASSWORD_LEN = 12


def main() -> None:
    try:
        email = TypeAdapter(EmailStr).validate_python(input("Admin email: ").strip())
    except ValidationError:
        sys.exit("Invalid email address.")

    # getpass hides what you type
    password = getpass.getpass(f"Password (min {MIN_PASSWORD_LEN} chars): ")
    if len(password) < MIN_PASSWORD_LEN or len(password) > 128:
        sys.exit(f"Password must be {MIN_PASSWORD_LEN}-128 characters.")
    if password != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")

    with SessionLocal() as db:
        if db.scalar(select(User).where(func.lower(User.email) == email.lower())):
            sys.exit("A user with that email already exists.")
        user = User(email=email, password_hash=hash_password(password), role=Role.ADMIN)
        db.add(user)
        db.flush()
        db.add(AuditLog(action="user_created", target_type="user", target_id=str(user.id),
                        details={"role": "admin", "via": "create_admin_script"}))
        db.commit()
        print(f"Admin created: {email}")


if __name__ == "__main__":
    main()