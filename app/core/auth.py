"""JWT authentication: token creation, verification, and FastAPI dependency."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import bcrypt
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.services.database import get_db_session

logger = structlog.get_logger(__name__)

_bearer = HTTPBearer(auto_error=True)


# ─── Password ─────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ─── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(user_id: UUID, session_id: UUID) -> str:
    """Create a signed JWT carrying user_id and session_id."""
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "session_id": str(session_id),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        logger.warning("jwt_decode_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ─── FastAPI Dependency ───────────────────────────────────────────────────────

class CurrentUser:
    """Resolved from the Bearer token — injected into protected endpoints."""

    def __init__(self, user_id: UUID, session_id: UUID) -> None:
        self.user_id = user_id
        self.session_id = session_id


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    """FastAPI dependency: validates JWT and returns CurrentUser."""
    payload = decode_token(credentials.credentials)
    try:
        user_id = UUID(payload["sub"])
        session_id = UUID(payload["session_id"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token payload",
        ) from exc

    structlog.contextvars.bind_contextvars(
        user_id=str(user_id),
        session_id=str(session_id),
    )
    return CurrentUser(user_id=user_id, session_id=session_id)
