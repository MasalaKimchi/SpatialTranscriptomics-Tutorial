"""PyTorch device selection with CUDA / MPS / CPU auto-detection."""

from __future__ import annotations

import torch


def resolve_device(request: str | torch.device | None = "auto") -> torch.device:
    """Pick the best available device.

    Priority when ``request`` is ``auto`` (default): CUDA → MPS → CPU.
    Explicit ``cuda``, ``mps``, or ``cpu`` is honored when available; otherwise
    falls back through the same priority chain with a warning printed once.
    """
    if isinstance(request, torch.device):
        return request

    req = (request or "auto").lower()
    if req not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            f"Unknown device {request!r}; use auto, cuda, mps, or cpu."
        )

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    if req == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        print("CUDA requested but unavailable; falling back to auto detection.")
        req = "auto"
    elif req == "mps":
        if mps_ok:
            return torch.device("mps")
        print("MPS requested but unavailable; falling back to auto detection.")
        req = "auto"
    elif req == "cpu":
        return torch.device("cpu")

    if cuda_ok:
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device | str) -> str:
    """Human-readable device string for logging."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is not None:
        name = torch.cuda.get_device_name(d)
        return f"cuda ({name})"
    if d.type == "mps":
        return "mps (Apple GPU)"
    return str(d)
