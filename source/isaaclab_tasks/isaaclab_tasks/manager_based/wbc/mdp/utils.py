from __future__ import annotations

from .commands import MotionCommand


def get_body_indices(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Get the indices of the bodies in the command.

    Args:
        command: The command to get the body indices from.
        body_names: The names of the bodies to get the indices for.

    Returns:
        The indices of the bodies in the command.
    """
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]