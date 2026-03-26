from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from multifit_optveri.math_utils import format_ratio

# Section 5 in the paper only fixes these acceleration cases for rho = 20/17
# and m in {8, ..., 12}. If you are checking paper/code alignment, start here
# to confirm the top-level case partition on p_n is identical.
PAPER_TARGET_RATIO = Fraction(20, 17)
PAPER_MACHINE_RANGE = range(8, 13)


@dataclass(frozen=True)
class PnRange:
    """Closed interval used by the implementation to represent a p_n case range.

    Note that the paper sometimes states strict upper bounds; when those bounds
    are used in the optimization model they must later be encoded as non-strict
    inequalities because Gurobi does not support strict constraints.
    """

    lower: Fraction | None = None
    upper: Fraction | None = None

    @property
    def text(self) -> str:
        """Human-readable rendering of the interval, mainly for plans/logs."""

        if self.lower is None and self.upper is None:
            return "unrestricted"
        if self.lower is not None and self.upper is None:
            return f"p_n >= {format_ratio(self.lower)}"
        if self.lower is None and self.upper is not None:
            return f"p_n <= {format_ratio(self.upper)}"
        return f"{format_ratio(self.lower)} <= p_n <= {format_ratio(self.upper)}"


class AccelerationCase(str, Enum):
    """Top-level acceleration branches from Section 5."""

    BASE = "base"
    CASE_1 = "case_1"
    CASE_2 = "case_2"
    CASE_3_1 = "case_3_1"
    CASE_3_2 = "case_3_2"

    @property
    def pn_range(self) -> PnRange:
        # This is the direct code translation of the paper's top-level partition
        # on p_n before any ell/profile branching is applied.
        if self is AccelerationCase.BASE:
            return PnRange()
        if self is AccelerationCase.CASE_1:
            return PnRange(lower=Fraction(11, 51))
        if self is AccelerationCase.CASE_2:
            return PnRange(lower=Fraction(7, 34), upper=Fraction(11, 51))
        return PnRange(upper=Fraction(7, 34))

    @property
    def uses_paper_acceleration(self) -> bool:
        """Whether this branch should trigger Section 5 cuts."""

        return self is not AccelerationCase.BASE


def parse_acceleration_case(value: str | AccelerationCase) -> AccelerationCase:
    """Parse CLI/config input into a normalized acceleration case enum."""

    if isinstance(value, AccelerationCase):
        return value

    normalized = value.strip().lower().replace("-", "_")
    try:
        return AccelerationCase(normalized)
    except ValueError as exc:
        supported = ", ".join(case.value for case in AccelerationCase)
        raise ValueError(
            f"Unsupported acceleration case '{value}'. Expected one of: {supported}."
        ) from exc


def paper_common_pn_lower_bound(machine_count: int) -> Fraction:
    # This is the common lower bound on p_n used before the case split in
    # Section 5. Compare this with the observation right before Case 1/2/3.
    if machine_count not in PAPER_MACHINE_RANGE:
        raise ValueError(
            f"Paper acceleration bounds are only defined for m in {PAPER_MACHINE_RANGE.start}.."
            f"{PAPER_MACHINE_RANGE.stop - 1}."
        )
    return Fraction(3 * machine_count, 17 * (machine_count - 1))
