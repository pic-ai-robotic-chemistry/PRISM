"""Checkpoint and normalization settings for CPL inverse design in the red phase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
_TOL = 1e-9


class ProfileError(ValueError):
    """Raised when a parameter is outside the model inputs."""


@dataclass(frozen=True)
class Range:
    """Closed interval used to normalize a model input."""

    min: float
    max: float

    @property
    def span(self) -> float:
        return self.max - self.min

    def contains(self, value: float) -> bool:
        return self.min - _TOL <= value <= self.max + _TOL

    def as_list(self) -> list[float]:
        return [self.min, self.max]


@dataclass(frozen=True)
class RedPhaseProfile:
    """Search and training settings for the bundled checkpoint."""

    name: str = "red"
    checkpoint: Path = SKILL_ROOT / "assets" / "best.ckpt"
    is_red: int = 1
    gray_norm: Range = Range(1.0, 5.0)
    thickness_norm: Range = Range(30.0, 80.0)
    stretch_norm: Range = Range(67.0, 200.0)
    gray_fixed: float = 5.0
    thickness_choices: tuple[float, ...] = (30.0, 48.0, 80.0)

    @property
    def param_ranges(self) -> dict[str, Range]:
        return {
            "gray": self.gray_norm,
            "thickness": self.thickness_norm,
            "stretch": self.stretch_norm,
        }

    def to_json(self) -> dict:
        return {
            "profile": self.name,
            "is_red": self.is_red,
            "checkpoint": str(self.checkpoint),
            "normalization": {name: bounds.as_list() for name, bounds in self.param_ranges.items()},
            "gray_search": [self.gray_fixed, self.gray_fixed],
            "gray_fixed": self.gray_fixed,
            "gray_choices": [self.gray_fixed],
            "thickness_choices": list(self.thickness_choices),
            "description": "Red phase with gray fixed at 5. The checkpoint was trained on all 270 samples.",
        }


RED_PHASE = RedPhaseProfile()


def normalize(param: str, value: float) -> float:
    """Min-max normalize a value against the checkpoint training range."""
    param_range = RED_PHASE.param_ranges.get(param)
    if param_range is None:
        raise ProfileError(f"Unknown model parameter {param!r}.")
    numeric = float(value)
    if not param_range.contains(numeric):
        raise ProfileError(f"{param}={numeric} is outside the training range {param_range.as_list()}.")
    return (numeric - param_range.min) / param_range.span
