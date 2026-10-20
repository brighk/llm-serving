"""Bearer-token authentication.

Tokens are 256-bit random strings, so a plain SHA-256 is enough to store them
safely (slow hashes like bcrypt exist to protect low-entropy passwords). Only
the hash is stored; the raw token is shown once, when it is issued.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import ApiToken, User, utcnow

TOKEN_PREFIX = "sk-llm-"


def generate_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_bearer(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if not hmac.compare_digest(scheme.lower(), "bearer") or not value:
        return None
    value = value.strip()
    if not value.startswith(TOKEN_PREFIX) or len(value) > 128:
        return None
    return value


async def authenticate(session: AsyncSession, raw_token: str | None) -> User | None:
    """Return the active user owning this token, or None."""
    if raw_token is None:
        return None
    result = await session.execute(
        select(ApiToken, User)
        .join(User, ApiToken.user_id == User.id)
        .where(ApiToken.token_hash == hash_token(raw_token), ApiToken.revoked.is_(False))
    )
    row = result.first()
    if row is None:
        return None
    token, user = row
    if user.disabled:
        return None
    token.last_used_at = utcnow()
    await session.commit()
    return user
