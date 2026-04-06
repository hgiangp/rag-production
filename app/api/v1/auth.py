"""Auth endpoints: register, login."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import create_access_token, hash_password, verify_password
from app.core.limiter import limiter
from app.models.session import Session
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.services.database import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
async def register(
    request: Request,
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    result = await db.exec(select(User).where(User.email == body.email))
    if result.first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.flush()

    session = Session(user_id=user.id, name="Default")
    db.add(session)
    await db.flush()

    token = create_access_token(user.id, session.id)
    logger.info("user_registered", user_id=str(user.id))
    return TokenResponse(access_token=token, session_id=str(session.id))


@router.post("/login", response_model=TokenResponse)
@limiter.limit("20/minute")
async def login(
    request: Request,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    result = await db.exec(select(User).where(User.email == body.email))
    user = result.first()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    session = Session(user_id=user.id)
    db.add(session)
    await db.flush()

    token = create_access_token(user.id, session.id)
    logger.info("user_logged_in", user_id=str(user.id), session_id=str(session.id))
    return TokenResponse(access_token=token, session_id=str(session.id))
