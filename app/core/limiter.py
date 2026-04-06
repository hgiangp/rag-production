"""Rate limiter instance and key extraction functions."""

from starlette.requests import Request

from slowapi import Limiter
from slowapi.util import get_remote_address as _get_remote_address


def get_user_or_ip(request: Request) -> str:
    """Use authenticated user ID as rate limit key; fall back to IP address."""
    user = getattr(request.state, "current_user", None)
    if user:
        return str(user.user_id)
    return _get_remote_address(request)


limiter = Limiter(key_func=get_user_or_ip, default_limits=["200 per day", "50 per hour"])
