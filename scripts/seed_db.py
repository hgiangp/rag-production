"""Seed the database with a default test user."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.core.config import settings
    from app.core.auth import hash_password
    from app.models.user import User
    from app.models.session import Session

    engine = create_async_engine(settings.postgres_dsn)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        user = User(email="test@example.com", hashed_password=hash_password("password123"))
        db.add(user)
        await db.flush()
        session = Session(user_id=user.id, name="Default Session")
        db.add(session)
        await db.commit()
        print(f"Created user: {user.email} (id={user.id})")
        print(f"Created session: {session.id}")


if __name__ == "__main__":
    asyncio.run(main())
