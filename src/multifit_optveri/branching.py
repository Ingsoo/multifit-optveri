from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

from multifit_optveri.acceleration import AccelerationCase

# This file is the executable version of the paper's outer verification loops:
# case -> m -> ell -> MTF profile -> OPT profile. When comparing with the paper,
# the most important question here is whether each iterator yields exactly the
# family of branches intended by the pseudocode and case analysis.


@dataclass(frozen=True)
class MtfProfile:
    """Coarse MTF machine-type profile used by the outer branch decomposition.

    The tuple fields correspond to consecutive machine blocks:
    (F1, R2, F2, R3, F3, R4, F4, R5).
    """

    m1f: int = 0
    m2r: int = 0
    m2f: int = 0
    m3r: int = 0
    m3f: int = 0
    m4r: int = 0
    m4f: int = 0
    m5r: int = 0

    @property
    def machine_count(self) -> int:
        """Total number of machines represented by this profile."""

        return self.m1f + self.m2r + self.m2f + self.m3r + self.m3f + self.m4r + self.m4f + self.m5r

    @property
    def scheduled_job_count(self) -> int:
        """Number of jobs that MTF successfully places before job n fails."""

        return (
            2 * (self.m1f + self.m2r)
            + 3 * (self.m2f + self.m3r)
            + 4 * (self.m3f + self.m4r)
            + 5 * (self.m4f + self.m5r)
        )

    @property
    def total_job_count(self) -> int:
        # In the verification setup, the last job n is the one that fails to fit in MTF.
        return self.scheduled_job_count + 1

    @property
    def compact_id(self) -> str:
        return f"mtf{self.m1f}{self.m2r}{self.m2f}{self.m3r}{self.m3f}{self.m4r}{self.m4f}{self.m5r}"

    @property
    def machine_cardinalities(self) -> tuple[int, ...]:
        """Per-machine job counts in the left-to-right block order."""

        return (
            (2,) * (self.m1f + self.m2r)
            + (3,) * (self.m2f + self.m3r)
            + (4,) * (self.m3f + self.m4r)
            + (5,) * (self.m4f + self.m5r)
        )

    @property
    def nF1(self) -> int:
        return self.m1f

    @property
    def nR2(self) -> int:
        return self.m2r

    @property
    def nF2(self) -> int:
        return self.m2f

    @property
    def nR3(self) -> int:
        return self.m3r

    @property
    def nF3(self) -> int:
        return self.m3f

    @property
    def nR4(self) -> int:
        return self.m4r

    @property
    def nF4(self) -> int:
        return self.m4f

    @property
    def nR5(self) -> int:
        return self.m5r

    @property
    def nM5(self) -> int:
        # Backward-compatible alias kept while the codebase transitions to R5.
        return self.nR5


@dataclass(frozen=True)
class OptProfile:
    """Coarse OPT machine-cardinality profile used in the outer branching."""

    m3: int
    m4: int
    m5: int
    pattern: str = "generic"

    @property
    def machine_count(self) -> int:
        """Total number of OPT machines represented by this profile."""

        return self.m3 + self.m4 + self.m5

    @property
    def total_job_count(self) -> int:
        """Total number of jobs implied by the OPT profile."""

        return 3 * self.m3 + 4 * self.m4 + 5 * self.m5

    @property
    def compact_id(self) -> str:
        # `pattern` is an implementation convenience for distinguishing cases such as
        # the even-ell Case 2 branch. The paper comparison should focus on the counts
        # (m3, m4, m5) first, then on whether the extra pattern-specific constraints
        # are added later in `models/obv.py`.
        suffix = self.pattern.replace("_", "")
        return f"opt{self.m3}{self.m4}{self.m5}{suffix}"

    @property
    def machine_cardinalities(self) -> tuple[int, ...]:
        """Per-machine OPT cardinalities in sorted machine order."""

        return (3,) * self.m3 + (4,) * self.m4 + (5,) * self.m5

    @property
    def nS3(self) -> int:
        return self.m3

    @property
    def nS4(self) -> int:
        return self.m4

    @property
    def nS5(self) -> int:
        return self.m5


@dataclass(frozen=True)
class FallbackStarts:
    """Starting indices of consecutive fallback blocks in F2/F3/F4."""

    s2: int | None = None
    s3: int | None = None
    s4: int | None = None

    @property
    def compact_id(self) -> str:
        def render(value: int | None) -> str:
            return "x" if value is None else f"{value:02d}"

        return f"fs{render(self.s2)}_{render(self.s3)}_{render(self.s4)}"


def ell_iterator(
    machine_count: int,
    acceleration_case: AccelerationCase,
    *,
    max_job_count: int,
) -> tuple[int, ...]:
    # Compare this function directly against the paper pseudocode's
    # `ell_iterator(m, case)`. If the branch order or feasible ell values differ,
    # Section 5 alignment is already broken before the model is built.

    # case 1. All odd ells from m+1 down to 1.
    if acceleration_case is AccelerationCase.CASE_1:
        return tuple(ell for ell in range(machine_count + 1, 0, -1) if ell % 2 == 1)

    if acceleration_case is AccelerationCase.CASE_2:
        upper = machine_count + 2 if machine_count % 2 == 0 else machine_count + 1
        return tuple(ell for ell in range(upper, 0, -1) if ell != 2)

    if acceleration_case is AccelerationCase.CASE_3_1:
        upper = machine_count + 3 if machine_count % 2 == 0 else machine_count + 2
        return tuple(range(upper, 0, -1))

    upper = machine_count + 1 if machine_count % 2 == 0 else machine_count + 2
    return tuple(range(upper, 3, -1))


def iter_opt_profiles(
    machine_count: int,
    ell: int,
    acceleration_case: AccelerationCase,
    *,
    mtf_profile: MtfProfile | None = None,
) -> Iterator[OptProfile]:
    # This is the outer OPT-profile branching logic implied by the case analysis.
    # It intentionally captures coarse profile families; the detailed structural
    # consequences of a profile are enforced later in `models/obv.py`.
    if acceleration_case is AccelerationCase.CASE_1:
        profile = OptProfile(m3=ell - 1, m4=machine_count - ell + 1, m5=0, pattern="case1")
        if mtf_profile is None or profile.total_job_count == mtf_profile.total_job_count:
            yield profile
        return

    if acceleration_case is AccelerationCase.CASE_2:
        if ell == 2:
            return
        if ell % 2 == 0:
            profile = OptProfile(
                m3=ell - 2,
                m4=machine_count - ell + 2,
                m5=0,
                pattern="two_long",
            )
        else:
            profile = OptProfile(
                m3=ell - 1,
                m4=machine_count - ell + 1,
                m5=0,
                pattern="regular",
            )
        if mtf_profile is None or profile.total_job_count == mtf_profile.total_job_count:
            yield profile
        return

    if acceleration_case is AccelerationCase.CASE_3:
        yield from _iter_case_3_opt_profiles(machine_count, mtf_profile=mtf_profile)
        return

    if acceleration_case is AccelerationCase.CASE_3_1:
        yield from _iter_case_3_1_opt_profiles(machine_count, ell, mtf_profile=mtf_profile)
        return

    yield from _iter_case_3_2_opt_profiles(machine_count, ell, mtf_profile=mtf_profile)


def _iter_case_3_opt_profiles(
    machine_count: int,
    *,
    mtf_profile: MtfProfile | None,
) -> Iterator[OptProfile]:
    for nS3, nS4, nS5 in product(range(machine_count + 1), repeat=3):
        if nS3 + nS4 + nS5 != machine_count:
            continue

        profile = OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
        if mtf_profile is None:
            yield profile
            continue
        if profile.total_job_count != mtf_profile.total_job_count:
            continue
        if _case_3_opt_profile_matches(mtf_profile, profile):
            yield profile


def _iter_case_3_1_opt_profiles(
    machine_count: int,
    ell: int,
    *,
    mtf_profile: MtfProfile | None,
) -> Iterator[OptProfile]:
    for nS3, nS4, nS5 in product(range(machine_count + 1), repeat=3):
        if nS3 + nS4 + nS5 != machine_count:
            continue

        profile = OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
        if mtf_profile is not None:
            if profile.total_job_count != mtf_profile.total_job_count:
                continue
            if _case_3_1_opt_profile_matches(ell, mtf_profile, profile):
                yield profile
            continue

        if ell in (1, 2):
            if nS3 == 0:
                yield profile
        elif ell % 2 == 1:
            if nS3 <= ell - 1:
                yield profile
        else:
            if nS3 <= ell - 2:
                yield profile


def _iter_case_3_2_opt_profiles(
    machine_count: int,
    ell: int,
    *,
    mtf_profile: MtfProfile | None,
) -> Iterator[OptProfile]:
    for nS3, nS4, nS5 in product(range(machine_count + 1), repeat=3):
        if nS3 + nS4 + nS5 != machine_count:
            continue

        profile = OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
        if mtf_profile is not None:
            if profile.total_job_count != mtf_profile.total_job_count:
                continue
            if _case_3_2_opt_profile_matches(ell, mtf_profile, profile):
                yield profile
            continue

        if nS4 < 1:
            continue
        if ell % 2 == 1:
            if nS3 <= ell - 5:
                yield profile
        else:
            if nS3 <= ell - 4:
                yield profile


def iter_mtf_profiles(
    machine_count: int,
    acceleration_case: AccelerationCase,
    *,
    min_job_count: int | None = None,
    max_job_count: int | None = None,
) -> Iterator[MtfProfile]:
    # MTF profile generation is independent of ell; candidate ell values are
    # recovered later by `candidate_ells_for_mtf_profile`.
    yield from _iter_all_mtf_profiles(
        machine_count,
        acceleration_case,
        min_job_count=min_job_count,
        max_job_count=max_job_count,
    )


def candidate_ells_for_mtf_profile(
    machine_count: int,
    mtf_profile: MtfProfile,
    acceleration_case: AccelerationCase,
) -> tuple[int, ...]:
    """Return the feasible ell values induced by one MTF profile.

    This is the inverse view of the paper's branching logic. In the current
    implementation it is mainly used to enumerate in the order
    m -> MTF profile -> candidate ells -> OPT profile.
    """

    prefix_total = mtf_profile.nF1 + mtf_profile.nR2 + mtf_profile.nF2
    # if prefix_total > machine_count // 2:
    #     return ()

    if acceleration_case is AccelerationCase.CASE_1:
        if mtf_profile.nF1 != 0:
            return ()
        return (2 * prefix_total + 1,)

    if acceleration_case is AccelerationCase.CASE_2:
        if mtf_profile.nF1 != 0:
            return ()
        return tuple(ell for ell in (2 * prefix_total + 1, 2 * prefix_total + 2) if ell != 2)

    if acceleration_case is AccelerationCase.CASE_3:
        return (
            2 * prefix_total + 1,
            2 * prefix_total + 2,
            2 * prefix_total + 3,
        )

    if acceleration_case is AccelerationCase.CASE_3_1:
        return (
            2 * prefix_total + 1,
            2 * prefix_total + 2,
            2 * prefix_total + 3,
        )

    if mtf_profile.nF2 == 0:
        return ()
    return (2 * prefix_total + 2, 2 * prefix_total + 3)


def iter_fallback_starts(
    machine_count: int,
    mtf_profile: MtfProfile,
    acceleration_case: AccelerationCase,
) -> Iterator[FallbackStarts]:
    """Enumerate coarse fallback-block starts once the MTF profile is fixed.

    Case 1 is left unbranched. For Case 2 we use the structural properties that
    fallback jobs in F2/F3/F4 are consecutive within each block and appear in
    block order F2 -> F3 -> F4 with no reversals.
    """

    if acceleration_case is AccelerationCase.CASE_1:
        yield FallbackStarts()
        return

    if acceleration_case in (
        AccelerationCase.CASE_2,
        AccelerationCase.CASE_3,
        AccelerationCase.CASE_3_1,
        AccelerationCase.CASE_3_2,
    ):
        scheduled_job_count = mtf_profile.scheduled_job_count
        structural_mins = fallback_start_structural_mins(machine_count, mtf_profile, acceleration_case)
        s2_structural_min = structural_mins.s2
        s3_structural_min = structural_mins.s3
        s4_structural_min = structural_mins.s4

        s2_values: tuple[int | None, ...]
        if mtf_profile.nF2 == 0:
            s2_values = (None,)
        else:
            assert s2_structural_min is not None
            s2_min = s2_structural_min
            s2_max = scheduled_job_count - (mtf_profile.nF2 + mtf_profile.nF3 + mtf_profile.nF4) + 1
            s2_values = tuple(range(s2_min, s2_max + 1))

        for s2 in s2_values:
            if mtf_profile.nF3 == 0:
                s3_values = (None,)
            else:
                assert s3_structural_min is not None
                s3_min = max(
                    (0 if s2 is None else s2 + mtf_profile.nF2),
                    s3_structural_min,
                )
                s3_max = scheduled_job_count - (mtf_profile.nF3 + mtf_profile.nF4) + 1
                s3_values = tuple(range(s3_min, s3_max + 1))

            for s3 in s3_values:
                if mtf_profile.nF4 == 0:
                    yield FallbackStarts(s2=s2, s3=s3, s4=None)
                    continue

                previous_fallback_end = 0
                if s3 is not None:
                    previous_fallback_end = s3 + mtf_profile.nF3
                elif s2 is not None:
                    previous_fallback_end = s2 + mtf_profile.nF2
                assert s4_structural_min is not None
                s4_min = max(
                    previous_fallback_end,
                    s4_structural_min,
                )
                s4_max = scheduled_job_count - mtf_profile.nF4 + 1
                for s4 in range(s4_min, s4_max + 1):
                    yield FallbackStarts(s2=s2, s3=s3, s4=s4)
        return


def fallback_start_structural_mins(
    machine_count: int,
    mtf_profile: MtfProfile,
    acceleration_case: AccelerationCase,
) -> FallbackStarts:
    """Return the current structural lower bounds for fallback block starts."""

    if acceleration_case is AccelerationCase.CASE_2:
        prefix_total = mtf_profile.nF1 + mtf_profile.nR2 + mtf_profile.nF2
        s2_min = None
        s3_min = None
        s4_min = None

        if mtf_profile.nF2 > 0:
            s2_min = 2 * prefix_total + 4
        if mtf_profile.nF3 > 0:
            s3_min = 2 * prefix_total + mtf_profile.nF2 + 3 * mtf_profile.nR3 + 3 * mtf_profile.nF3 + 2
        if mtf_profile.nF4 > 0:
            s4_min = (
                2 * prefix_total
                + mtf_profile.nF2
                + 3 * mtf_profile.nR3
                + 4 * mtf_profile.nF3
                + 4 * mtf_profile.nR4
                + 4 * mtf_profile.nF4
                + 2
            )
        return FallbackStarts(s2=s2_min, s3=s3_min, s4=s4_min)

    if acceleration_case in (
        AccelerationCase.CASE_3,
        AccelerationCase.CASE_3_1,
        AccelerationCase.CASE_3_2,
    ):
        prefix_total = mtf_profile.nF1 + mtf_profile.nR2 + mtf_profile.nF2
        s2_min = None
        s3_min = None
        s4_min = None

        if mtf_profile.nF2 > 0:
            s2_min = 2 * prefix_total + 2
        if mtf_profile.nF3 > 0:
            s3_min = 2 * prefix_total + mtf_profile.nF2 + 3 * mtf_profile.nR3 + 3 * mtf_profile.nF3 + 2
        if mtf_profile.nF4 > 0:
            s4_min = (
                2 * prefix_total
                + mtf_profile.nF2
                + 3 * mtf_profile.nR3
                + 4 * mtf_profile.nF3
                + 4 * mtf_profile.nR4
                + 4 * mtf_profile.nF4
                + 2
            )
        return FallbackStarts(s2=s2_min, s3=s3_min, s4=s4_min)

    return FallbackStarts()


def _iter_all_mtf_profiles(
    machine_count: int,
    acceleration_case: AccelerationCase,
    *,
    min_job_count: int | None = None,
    max_job_count: int | None = None,
) -> Iterator[MtfProfile]:
    # Generate coarse MTF profiles independent of ell; the possible ell values
    # are recovered later by `candidate_ells_for_mtf_profile`.
    if acceleration_case is AccelerationCase.CASE_1:
        pair_upper = machine_count // 2
        for pair_total in range(pair_upper + 1):
            tail_total = machine_count - pair_total
            target_job_count = 4 * machine_count - 2 * pair_total
            if min_job_count is not None and target_job_count < min_job_count:
                continue
            if max_job_count is not None and target_job_count > max_job_count:
                continue
            for nR2 in range(pair_total + 1):
                nF2 = pair_total - nR2
                for nF3 in range(tail_total + 1):
                    for nR4 in range(tail_total - nF3 + 1):
                        for nF4 in range(tail_total - nF3 - nR4 + 1):
                            for nR5 in range(tail_total - nF3 - nR4 - nF4 + 1):
                                nR3 = tail_total - nF3 - nR4 - nF4 - nR5
                                # Case 1 job-count identity rewritten in terms
                                # of the MTF profile variables:
                                # nF2 - nR3 + nF4 + nR5 + 1 = 0.
                                if nF2 - nR3 + nF4 + nR5 + 1 != 0:
                                    continue
                                profile = MtfProfile(0, nR2, nF2, nR3, nF3, nR4, nF4, nR5)
                                if profile.total_job_count == target_job_count:
                                    yield profile
        return

    if acceleration_case is AccelerationCase.CASE_2:
        m = machine_count
        max_nF3 = (2 * m - 12) // 7
        for pair_total in range(m // 2 + 1):
            # pair_total = nR2 + nF2
            for nR2 in range(pair_total + 1):
                nF2 = pair_total - nR2
                # From nR2 + 2*nR3 + nF3 + nR4 = m + 1 and nR4 >= 0,
                # we need 2*nR3 <= m + 1 - nR2 - nF3.
                for nF3 in range(max_nF3 + 1):
                    max_nR3 = (m + 1 - nR2 - nF3) // 2
                    if max_nR3 < 0:
                        continue
                    # Since nR3 - nF2 = nF4 + nR5 + 1 >= 1,
                    # we need nR3 >= nF2 + 1.
                    min_nR3 = nF2 + 1
                    if min_nR3 > max_nR3:
                        continue
                    for nR3 in range(min_nR3, max_nR3 + 1):
                        nR4 = m + 1 - nR2 - 2 * nR3 - nF3
                        if nR4 < 0:
                            continue
                        # nR5 = nR3 - nF2 - nF4 - 1 >= 0
                        # so nF4 <= nR3 - nF2 - 1
                        max_nF4_by_relation = nR3 - nF2 - 1
                        # 7*nF3 + 10*nF4 < 2*m - 11
                        max_nF4_by_ineq = (2 * m - 12 - 7 * nF3) // 10
                        max_nF4 = min(max_nF4_by_relation, max_nF4_by_ineq)
                        if max_nF4 < 0:
                            continue
                        for nF4 in range(max_nF4 + 1):
                            nR5 = nR3 - nF2 - nF4 - 1
                            if nF4 + nR5 + 1 != nR3 - nF2:
                                continue
                            if 7 * nF3 + 10 * nF4 >= 2 * m - 11:
                                continue
                            profile = MtfProfile(0, nR2, nF2, nR3, nF3, nR4, nF4, nR5)
                            if min_job_count is not None and profile.total_job_count < min_job_count:
                                continue
                            if max_job_count is not None and profile.total_job_count > max_job_count:
                                continue
                            yield profile
        return

    if acceleration_case is AccelerationCase.CASE_3:
        m = machine_count
        max_nF3 = max(0, (m - 8) // 4)
        for a in range(m + 1):
            for nF1 in range(a + 1):
                nR2 = a - nF1
                remaining_after_prefix = m - a
                for nF2 in range(remaining_after_prefix + 1):
                    remaining_after_f2 = remaining_after_prefix - nF2
                    for nR3 in range(remaining_after_f2 + 1):
                        remaining_after_r3 = remaining_after_f2 - nR3
                        for nF3 in range(min(max_nF3, remaining_after_r3) + 1):
                            remaining_tail = remaining_after_r3 - nF3
                            for nR5 in range(remaining_tail + 1):
                                nR4 = remaining_tail - nR5
                                profile = MtfProfile(nF1, nR2, nF2, nR3, nF3, nR4, 0, nR5)
                                if min_job_count is not None and profile.total_job_count < min_job_count:
                                    continue
                                if max_job_count is not None and profile.total_job_count > max_job_count:
                                    continue
                                yield profile
        return

    if acceleration_case is AccelerationCase.CASE_3_1:
        yield from _iter_all_case_3_profiles(
            machine_count,
            allow_case_32=False,
            min_job_count=min_job_count,
            max_job_count=max_job_count,
        )
        return

    yield from _iter_all_case_3_profiles(
        machine_count,
        allow_case_32=True,
        min_job_count=min_job_count,
        max_job_count=max_job_count,
    )


def _iter_all_case_3_profiles(
    machine_count: int,
    *,
    allow_case_32: bool,
    min_job_count: int | None = None,
    max_job_count: int | None = None,
) -> Iterator[MtfProfile]:
    # Case 3 is split into 3-1 / 3-2 by whether ell is a fallback job in F2.
    # `allow_case_32` toggles that branch. This helper is a good place to audit
    # whether the code follows the paper's intended admissible profile region.
    f3_upper = max(0, (3 * machine_count - 22) // 12)
    for a in range(machine_count + 1):
        # `a` is the total size of the F1/R2 prefix in terms of machines.
        for nF1 in range(a + 1):
            nR2 = a - nF1
            for nF2 in range(machine_count - a + 1):
                if a + nF2 > machine_count // 2:
                    continue
                if allow_case_32 and nF2 == 0:
                    continue
                for b in range(nF2, machine_count - a + 1):
                    nR3 = b - nF2
                    for nF3 in range(f3_upper + 1):
                        if 2 * nF2 + 12 * nF3 >= 3 * machine_count - 21:
                            continue
                        for nR5 in range(machine_count - a - b - nF3 + 1):
                            nR4 = machine_count - a - b - nF3 - nR5
                            profile = MtfProfile(nF1, nR2, nF2, nR3, nF3, nR4, 0, nR5)
                            if min_job_count is not None and profile.total_job_count < min_job_count:
                                continue
                            if max_job_count is not None and profile.total_job_count > max_job_count:
                                continue
                            yield profile


def _case_3_1_opt_profile_matches(
    ell: int,
    mtf_profile: MtfProfile,
    opt_profile: OptProfile,
) -> bool:
    a = mtf_profile.nF1 + mtf_profile.nR2
    nF2 = mtf_profile.nF2
    nS3 = opt_profile.nS3

    if not (2 * a - 1 <= nS3 <= 2 * (a + nF2)):
        return False
    if ell in (1, 2):
        return nS3 == 0
    if ell % 2 == 1:
        return nS3 <= ell - 1
    return nS3 <= ell - 2


def _case_3_opt_profile_matches(
    mtf_profile: MtfProfile,
    opt_profile: OptProfile,
) -> bool:
    a = mtf_profile.nF1 + mtf_profile.nR2
    return opt_profile.nS3 >= 2 * a - 1


def _case_3_2_opt_profile_matches(
    ell: int,
    mtf_profile: MtfProfile,
    opt_profile: OptProfile,
) -> bool:
    a = mtf_profile.nF1 + mtf_profile.nR2
    nF2 = mtf_profile.nF2
    nS3 = opt_profile.nS3
    nS4 = opt_profile.nS4

    if nF2 == 0 or nS4 < 1 or nS4 < 2 * nF2 - 1:
        return False
    if not (2 * a - 1 <= nS3 <= 2 * a):
        return False
    if ell % 2 == 1:
        return nS3 <= ell - 5
    return nS3 <= ell - 4
