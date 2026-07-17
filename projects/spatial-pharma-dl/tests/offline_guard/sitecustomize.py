"""Fail-closed socket guard inherited by offline Python subprocesses."""

from __future__ import annotations

import os
import socket


class OfflineNetworkError(RuntimeError):
    """Raised when an offline Python child attempts network access."""


def _deny_socket(*_args: object, **_kwargs: object) -> None:
    raise OfflineNetworkError(
        "Network access is disabled for the offline evidence tier in this "
        "Python subprocess."
    )


if os.environ.get("SPATIAL_TX_OFFLINE_GUARD") == "1":
    socket.socket.connect = _deny_socket  # type: ignore[method-assign]
    socket.socket.connect_ex = _deny_socket  # type: ignore[method-assign]
    socket.create_connection = _deny_socket
    socket.getaddrinfo = _deny_socket
    socket.gethostbyname = _deny_socket
    socket.gethostbyname_ex = _deny_socket
    socket.gethostbyaddr = _deny_socket
