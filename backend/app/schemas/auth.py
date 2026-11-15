import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import Role


class LoginRequest(BaseModel):
    # Unknown extra fields in the request are rejected
    model_config = ConfigDict(extra="forbid")

    # Must be a valid email address
    email: EmailStr
    # The maximum length stops someone from sending a huge "password" to burn CPU in argon2
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    # Lets this be built directly from a database User object
    model_config = ConfigDict(from_attributes=True)

    # Only these fields are ever sent to the browser. password_hash is never included.
    id: uuid.UUID
    email: str
    role: Role
    employee_id: uuid.UUID | None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the access token expires
    user: UserOut