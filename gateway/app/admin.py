"""Admin CLI. Deliberately not an HTTP API: admin actions need `kubectl exec`
into the gateway pod, so there is no admin endpoint to attack.

  python -m app.admin create-user alice --limit 200000
  python -m app.admin issue-token alice          # prints the token ONCE
  python -m app.admin revoke-token sk-llm-AbCd
  python -m app.admin set-limit alice 500000
  python -m app.admin reset-usage alice
  python -m app.admin disable alice              # the reaper kills the pod
  python -m app.admin list
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import generate_token, hash_token
from .config import Settings
from .db import ApiToken, User, init_db, make_engine, make_sessionmaker


async def get_user(session: AsyncSession, name: str) -> User:
    user = (await session.execute(select(User).where(User.name == name))).scalar_one_or_none()
    if user is None:
        raise SystemExit(f"no such user: {name}")
    return user


async def create_user(session: AsyncSession, name: str, limit: int) -> User:
    user = User(name=name, token_limit=limit)
    session.add(user)
    await session.commit()
    return user


async def issue_token(session: AsyncSession, name: str) -> str:
    user = await get_user(session, name)
    raw = generate_token()
    session.add(ApiToken(user_id=user.id, token_hash=hash_token(raw), prefix=raw[:11]))
    await session.commit()
    return raw


async def revoke_token(session: AsyncSession, prefix: str) -> int:
    result = await session.execute(
        update(ApiToken).where(ApiToken.prefix == prefix).values(revoked=True)
    )
    await session.commit()
    return result.rowcount


async def _main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("create-user"); p.add_argument("name"); p.add_argument("--limit", type=int, default=100_000)
    p = sub.add_parser("issue-token"); p.add_argument("name")
    p = sub.add_parser("revoke-token"); p.add_argument("prefix")
    p = sub.add_parser("set-limit"); p.add_argument("name"); p.add_argument("limit", type=int)
    p = sub.add_parser("reset-usage"); p.add_argument("name")
    p = sub.add_parser("disable"); p.add_argument("name")
    p = sub.add_parser("enable"); p.add_argument("name")
    sub.add_parser("list")
    args = parser.parse_args(argv)

    engine = make_engine(Settings.from_env().database_url)
    await init_db(engine)
    sm = make_sessionmaker(engine)
    try:
        async with sm() as s:
            if args.cmd == "create-user":
                u = await create_user(s, args.name, args.limit)
                print(f"created {u.name} id={u.id} limit={u.token_limit}")
            elif args.cmd == "issue-token":
                print(await issue_token(s, args.name))
                print("Store this token now; it cannot be shown again.", file=sys.stderr)
            elif args.cmd == "revoke-token":
                print(f"revoked {await revoke_token(s, args.prefix)} token(s)")
            elif args.cmd in ("set-limit", "reset-usage", "disable", "enable"):
                u = await get_user(s, args.name)
                if args.cmd == "set-limit":
                    u.token_limit = args.limit
                elif args.cmd == "reset-usage":
                    u.tokens_used = 0
                else:
                    u.disabled = args.cmd == "disable"
                await s.commit()
                print(f"{u.name}: used={u.tokens_used} limit={u.token_limit} disabled={u.disabled}")
            elif args.cmd == "list":
                for u in (await s.execute(select(User).order_by(User.name))).scalars():
                    print(f"{u.name:20} used={u.tokens_used:>10} limit={u.token_limit:>10} disabled={u.disabled}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1:]))
