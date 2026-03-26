from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

from multifit_optveri.acceleration import AccelerationCase


@dataclass(frozen=True)
class MtfProfile:
    m1f: int = 0
    m2r: int = 0
    m2f: int = 0
    m3r: int = 0
    m3f: int = 0
    m4: int = 0
    m5: int = 0

    @property
    def machine_count(self) -> int:
        return self.m1f + self.m2r + self.m2f + self.m3r + self.m3f + self.m4 + self.m5

    @property
    def scheduled_job_count(self) -> int:
        return (
            2 * (self.m1f + self.m2r)
            + 3 * (self.m2f + self.m3r)
            + 4 * (self.m3f + self.m4)
            + 5 * self.m5
        )

    @property
    def total_job_count(self) -> int:
        # In the verification setup, the last job n is the one that fails to fit in MTF.
        return self.scheduled_job_count + 1

    @property
    def compact_id(self) -> str:
        return (
            f"mtf{self.m1f}{self.m2r}{self.m2f}{self.m3r}{self.m3f}{self.m4}{self.m5}"
        )

    @property
    def machine_cardinalities(self) -> tuple[int, ...]:
        return (
            (2,) * (self.m1f + self.m2r)
            + (3,) * (self.m2f + self.m3r)
            + (4,) * (self.m3f + self.m4)
            + (5,) * self.m5
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
        return self.m4

    @property
    def nM5(self) -> int:
        return self.m5


@dataclass(frozen=True)
class OptProfile:
    m3: int
    m4: int
    m5: int
    pattern: str = "generic"

    @property
    def machine_count(self) -> int:
        return self.m3 + self.m4 + self.m5

    @property
    def total_job_count(self) -> int:
        return 3 * self.m3 + 4 * self.m4 + 5 * self.m5

    @property
    def compact_id(self) -> str:
        suffix = self.pattern.replace("_", "")
        return f"opt{self.m3}{self.m4}{self.m5}{suffix}"

    @property
    def machine_cardinalities(self) -> tuple[int, ...]:
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


def ell_iterator(
    machine_count: int,
    acceleration_case: AccelerationCase,
    *,
    max_job_count: int,
) -> tuple[int, ...]:
    if acceleration_case is AccelerationCase.CASE_1:
        return tuple(reversed(list(range(1, machine_count + 2, 2))))

    if acceleration_case is AccelerationCase.CASE_2:
        upper = machine_count + 3 if machine_count % 2 == 0 else machine_count + 2
        return tuple(reversed([ell for ell in range(1, upper) if ell != 2]))

    if acceleration_case is AccelerationCase.CASE_3_1:
        return tuple(reversed(list(range(1, machine_count + 4))))

    return tuple(reversed(list(range(4, machine_count + 5))))


def iter_opt_profiles(
    machine_count: int,
    ell: int,
    acceleration_case: AccelerationCase,
) -> Iterator[OptProfile]:
    if acceleration_case is AccelerationCase.CASE_1:
        yield OptProfile(m3=ell - 1, m4=machine_count - ell + 1, m5=0, pattern="case1")
        return

    if acceleration_case is AccelerationCase.CASE_2:
        if ell % 2 == 0:
            yield OptProfile(
                m3=ell - 2,
                m4=machine_count - ell + 2,
                m5=0,
                pattern="two_long",
            )
        else:
            yield OptProfile(
                m3=ell - 1,
                m4=machine_count - ell + 1,
                m5=0,
                pattern="regular",
            )
        return

    for nS3, nS4, nS5 in product(range(machine_count + 1), repeat=3):
        if nS3 + nS4 + nS5 != machine_count:
            continue

        if acceleration_case is AccelerationCase.CASE_3_1:
            if ell in (1, 2):
                if nS3 == 0:
                    yield OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
            elif ell % 2 == 1:
                if nS3 <= ell - 1:
                    yield OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
            else:
                if nS3 <= ell - 2:
                    yield OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
            continue

        if nS4 >= 1:
            if ell % 2 == 1:
                if nS3 <= ell - 5:
                    yield OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")
            else:
                if nS3 <= ell - 4:
                    yield OptProfile(m3=nS3, m4=nS4, m5=nS5, pattern="generic")


def iter_mtf_profiles(
    machine_count: int,
    ell: int,
    opt_profile: OptProfile,
    acceleration_case: AccelerationCase,
) -> Iterator[MtfProfile]:
    nS3, nS4, nS5 = opt_profile.nS3, opt_profile.nS4, opt_profile.nS5
    job_count = opt_profile.total_job_count

    if acceleration_case is AccelerationCase.CASE_1:
        pair_total = nS3 // 2
        if 2 * pair_total != nS3:
            return
        tail_total = machine_count - pair_total
        for nF2 in range(pair_total + 1):
            nR2 = pair_total - nF2
            for nF3 in range(tail_total + 1):
                for nR4 in range(tail_total - nF3 + 1):
                    for nM5 in range(tail_total - nF3 - nR4 + 1):
                        nR3 = tail_total - nF3 - nR4 - nM5
                        if (
                            2 * nR2
                            + 3 * (nF2 + nR3)
                            + 4 * (nF3 + nR4)
                            + 5 * nM5
                            + 1
                            == job_count
                        ):
                            yield MtfProfile(0, nR2, nF2, nR3, nF3, nR4, nM5)
        return

    if acceleration_case is AccelerationCase.CASE_2:
        pair_total = nS3 // 2
        if 2 * pair_total != nS3:
            return
        tail_total = machine_count - pair_total
        for nF2 in range(pair_total + 1):
            nR2 = pair_total - nF2
            for nF3 in range(tail_total + 1):
                for nR4 in range(tail_total - nF3 + 1):
                    for nM5 in range(tail_total - nF3 - nR4 + 1):
                        nR3 = tail_total - nF3 - nR4 - nM5
                        if all(
                            (
                                2 * nR2
                                + 3 * nF2
                                + 3 * nR3
                                + 4 * (nF3 + nR4)
                                + 5 * nM5
                                == job_count - 1,
                                2 * nF2 + 7 * nF3 < 2 * machine_count - 11,
                                6 * nM5 < 2 * machine_count - 11,
                            )
                        ):
                            yield MtfProfile(0, nR2, nF2, nR3, nF3, nR4, nM5)
        return

    if acceleration_case is AccelerationCase.CASE_3_1:
        yield from _iter_case_3_profiles(
            machine_count,
            ell,
            nS3,
            nS4,
            job_count,
            allow_case_32=False,
        )
        return

    yield from _iter_case_3_profiles(
        machine_count,
        ell,
        nS3,
        nS4,
        job_count,
        allow_case_32=True,
    )


def _iter_case_3_profiles(
    machine_count: int,
    ell: int,
    nS3: int,
    nS4: int,
    job_count: int,
    *,
    allow_case_32: bool,
) -> Iterator[MtfProfile]:
    f3_upper = max(0, (3 * machine_count - 22) // 12)
    for a in range(machine_count + 1):
        for nF1 in range(a + 1):
            nR2 = a - nF1
            if allow_case_32:
                if not (2 * a - 1 <= nS3 <= 2 * a):
                    continue
            else:
                if not (2 * a - 1 <= nS3):
                    continue
            for nF2 in range(machine_count - a + 1):
                if allow_case_32:
                    if nF2 == 0 or nS4 < 2 * nF2 - 1:
                        continue
                if not allow_case_32 and nS3 > 2 * (a + nF2):
                    continue
                if allow_case_32:
                    if not (2 * (a + nF2) + 2 <= ell <= 2 * (a + nF2) + 3):
                        continue
                else:
                    if ell - 3 == 2 * (a + nF2) and nF2 != 0:
                        continue
                for b in range(nF2, machine_count - a + 1):
                    nR3 = b - nF2
                    nM5 = job_count - 1 - 4 * machine_count + 2 * a + b
                    if nM5 < 0:
                        continue
                    for nF3 in range(f3_upper + 1):
                        if 2 * nF2 + 12 * nF3 >= 3 * machine_count - 21:
                            continue
                        nR4 = machine_count - a - b - nF3 - nM5
                        if nR4 < 0:
                            continue
                        yield MtfProfile(nF1, nR2, nF2, nR3, nF3, nR4, nM5)
