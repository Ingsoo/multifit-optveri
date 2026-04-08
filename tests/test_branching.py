from __future__ import annotations

import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import (
    FallbackStarts,
    MtfProfile,
    OptProfile,
    candidate_ells_for_mtf_profile,
    ell_iterator,
    iter_fallback_starts,
    iter_mtf_profiles,
    iter_opt_profiles,
)


def _legacy_case_2_profiles(machine_count: int, *, max_job_count: int | None = None) -> set[MtfProfile]:
    profiles: set[MtfProfile] = set()
    pair_upper = machine_count // 2
    for pair_total in range(1, pair_upper + 1):
        tail_total = machine_count - pair_total
        for nR2 in range(pair_total + 1):
            nF2 = pair_total - nR2
            for nF3 in range(tail_total + 1):
                for nR4 in range(tail_total - nF3 + 1):
                    for nF4 in range(tail_total - nF3 - nR4 + 1):
                        remainder = tail_total - nF3 - nR4 - nF4
                        if (remainder + nF2 + nF4 + 1) % 2 != 0:
                            continue
                        nR3 = (remainder + nF2 + nF4 + 1) // 2
                        nR5 = nR3 - nF2 - nF4 - 1
                        if nR5 < 0 or nR3 < 0:
                            continue
                        if nR3 + nR5 != remainder:
                            continue
                        if nF4 + nR5 + 1 != nR3 - nF2:
                            continue
                        if 7 * nF3 + 10 * nF4 < 2 * machine_count - 11:
                            profile = MtfProfile(0, nR2, nF2, nR3, nF3, nR4, nF4, nR5)
                            if max_job_count is None or profile.total_job_count <= max_job_count:
                                profiles.add(profile)
    return profiles


class BranchingTests(unittest.TestCase):
    def test_mtf_profile_properties(self) -> None:
        profile = MtfProfile(1, 2, 1, 1, 1, 1, 0, 1)

        self.assertEqual(profile.machine_count, 8)
        self.assertEqual(profile.scheduled_job_count, 25)
        self.assertEqual(profile.total_job_count, 26)
        self.assertEqual(profile.compact_id, "mtf12111101")
        self.assertEqual(profile.machine_cardinalities, (2, 2, 2, 3, 3, 4, 4, 5))
        self.assertEqual(profile.nF1, 1)
        self.assertEqual(profile.nR4, 1)
        self.assertEqual(profile.nF4, 0)
        self.assertEqual(profile.nR5, 1)

    def test_opt_profile_properties(self) -> None:
        profile = OptProfile(3, 4, 1, pattern="two_long")

        self.assertEqual(profile.machine_count, 8)
        self.assertEqual(profile.total_job_count, 30)
        self.assertEqual(profile.compact_id, "opt341twolong")
        self.assertEqual(profile.machine_cardinalities, (3, 3, 3, 4, 4, 4, 4, 5))
        self.assertEqual(profile.nS3, 3)
        self.assertEqual(profile.nS5, 1)

    def test_fallback_starts_compact_id(self) -> None:
        self.assertEqual(FallbackStarts().compact_id, "fsx_x_x")
        self.assertEqual(FallbackStarts(12, 18, None).compact_id, "fs12_18_x")

    def test_ell_iterator_matches_paper_order_for_m8(self) -> None:
        self.assertEqual(ell_iterator(8, AccelerationCase.CASE_1, max_job_count=100), (9, 7, 5, 3, 1))
        self.assertEqual(ell_iterator(8, AccelerationCase.CASE_2, max_job_count=100), (10, 9, 8, 7, 6, 5, 4, 3, 1))
        self.assertEqual(ell_iterator(8, AccelerationCase.CASE_3_1, max_job_count=100), tuple(range(11, 0, -1)))
        self.assertEqual(ell_iterator(8, AccelerationCase.CASE_3_2, max_job_count=100), tuple(range(9, 3, -1)))

    def test_case_1_and_case_2_opt_profiles_are_fully_determined(self) -> None:
        self.assertEqual(
            list(iter_opt_profiles(8, 9, AccelerationCase.CASE_1)),
            [OptProfile(8, 0, 0, pattern="case1")],
        )
        self.assertEqual(list(iter_opt_profiles(8, 2, AccelerationCase.CASE_2)), [])
        self.assertEqual(
            list(iter_opt_profiles(8, 10, AccelerationCase.CASE_2)),
            [OptProfile(8, 0, 0, pattern="two_long")],
        )
        self.assertEqual(
            list(iter_opt_profiles(8, 9, AccelerationCase.CASE_2)),
            [OptProfile(8, 0, 0, pattern="regular")],
        )

    def test_case_3_opt_profiles_only_enforce_ns3_prefix_lower_bound(self) -> None:
        mtf_profile = MtfProfile(1, 0, 0, 0, 0, 6, 0, 1)

        profiles = list(iter_opt_profiles(8, 3, AccelerationCase.CASE_3, mtf_profile=mtf_profile))

        self.assertIn(OptProfile(1, 6, 1, pattern="generic"), profiles)
        self.assertNotIn(OptProfile(0, 8, 0, pattern="generic"), profiles)
        self.assertTrue(all(profile.nS3 >= 1 for profile in profiles))

    def test_candidate_ells_for_mtf_profile_match_case_logic(self) -> None:
        self.assertEqual(
            candidate_ells_for_mtf_profile(
                8,
                MtfProfile(0, 0, 0, 0, 0, 0, 0, 8),
                AccelerationCase.CASE_2,
            ),
            (1,),
        )
        self.assertEqual(
            candidate_ells_for_mtf_profile(
                8,
                MtfProfile(0, 4, 0, 1, 0, 3, 0, 0),
                AccelerationCase.CASE_2,
            ),
            (9, 10),
        )
        self.assertEqual(
            candidate_ells_for_mtf_profile(
                8,
                MtfProfile(0, 0, 1, 0, 0, 7, 0, 0),
                AccelerationCase.CASE_3_1,
            ),
            (3, 4, 5),
        )
        self.assertEqual(
            candidate_ells_for_mtf_profile(
                8,
                MtfProfile(0, 0, 1, 0, 0, 0, 0, 7),
                AccelerationCase.CASE_3_2,
            ),
            (4, 5),
        )

    def test_case_1_and_case_2_mtf_profile_counts_match_current_branching(self) -> None:
        raw_case_1_profiles = list(iter_mtf_profiles(8, AccelerationCase.CASE_1, max_job_count=100))
        case_1_profiles = list(
            profile
            for profile in raw_case_1_profiles
            if 9 in candidate_ells_for_mtf_profile(8, profile, AccelerationCase.CASE_1)
            and any(
                candidate == OptProfile(8, 0, 0, pattern="case1")
                for candidate in iter_opt_profiles(
                    8,
                    9,
                    AccelerationCase.CASE_1,
                    mtf_profile=profile,
                )
            )
        )
        case_2_profiles = list(
            profile
            for profile in iter_mtf_profiles(8, AccelerationCase.CASE_2, max_job_count=100)
            if 10 in candidate_ells_for_mtf_profile(8, profile, AccelerationCase.CASE_2)
            and any(
                candidate == OptProfile(8, 0, 0, pattern="two_long")
                for candidate in iter_opt_profiles(
                    8,
                    10,
                    AccelerationCase.CASE_2,
                    mtf_profile=profile,
                )
            )
        )

        self.assertTrue(all(profile.total_job_count >= 24 for profile in raw_case_1_profiles))
        self.assertTrue(all(profile.total_job_count == 24 for profile in case_1_profiles))
        self.assertEqual(len(case_1_profiles), 16)
        self.assertEqual(
            [profile.compact_id for profile in case_1_profiles[:3]],
            ["mtf01340000", "mtf02230100", "mtf02231000"],
        )
        self.assertEqual(len(case_2_profiles), 6)
        self.assertEqual(
            [profile.compact_id for profile in case_2_profiles],
            [
                "mtf01340000",
                "mtf02230100",
                "mtf03120200",
                "mtf03130001",
                "mtf04010300",
                "mtf04020101",
            ],
        )

    def test_case_3_profiles_satisfy_basic_invariants(self) -> None:
        case_31_profiles = list(
            profile
            for profile in iter_mtf_profiles(8, AccelerationCase.CASE_3_1, max_job_count=32)
            if 4 in candidate_ells_for_mtf_profile(8, profile, AccelerationCase.CASE_3_1)
            and any(
                candidate == OptProfile(0, 8, 0, pattern="generic")
                for candidate in iter_opt_profiles(
                    8,
                    4,
                    AccelerationCase.CASE_3_1,
                    mtf_profile=profile,
                )
            )
        )
        case_32_profiles = list(
            profile
            for profile in iter_mtf_profiles(8, AccelerationCase.CASE_3_2, max_job_count=39)
            if 4 in candidate_ells_for_mtf_profile(8, profile, AccelerationCase.CASE_3_2)
            and any(
                candidate == OptProfile(0, 1, 7, pattern="generic")
                for candidate in iter_opt_profiles(
                    8,
                    4,
                    AccelerationCase.CASE_3_2,
                    mtf_profile=profile,
                )
            )
        )

        self.assertEqual(
            [profile.compact_id for profile in case_31_profiles],
            [
                "mtf00100700",
                "mtf00110501",
                "mtf00120302",
                "mtf00130103",
            ],
        )
        self.assertEqual([profile.compact_id for profile in case_32_profiles], ["mtf00100007"])

        for profile in case_31_profiles + case_32_profiles:
            self.assertEqual(profile.machine_count, 8)

    def test_case_2_profile_space_can_explicitly_include_f4(self) -> None:
        profiles = list(iter_mtf_profiles(12, AccelerationCase.CASE_2, max_job_count=200))

        self.assertTrue(any(profile.nF4 > 0 for profile in profiles))

    def test_case_3_profiles_only_enforce_f3_bound_and_no_f4(self) -> None:
        profiles = list(iter_mtf_profiles(8, AccelerationCase.CASE_3, max_job_count=200))

        self.assertTrue(profiles)
        self.assertTrue(all(profile.nF4 == 0 for profile in profiles))
        self.assertTrue(all(4 * profile.nF3 < 8 - 7 for profile in profiles))

    def test_iter_mtf_profiles_respects_min_job_count(self) -> None:
        profiles = list(iter_mtf_profiles(8, AccelerationCase.CASE_3, min_job_count=31, max_job_count=200))

        self.assertTrue(profiles)
        self.assertTrue(all(profile.total_job_count >= 31 for profile in profiles))

    def test_case_2_new_generation_matches_legacy_generation_when_pair_total_positive(self) -> None:
        for machine_count in range(8, 13):
            new_profiles = {
                profile
                for profile in iter_mtf_profiles(machine_count, AccelerationCase.CASE_2, max_job_count=200)
                if profile.nR2 + profile.nF2 >= 1
            }
            legacy_profiles = _legacy_case_2_profiles(machine_count, max_job_count=200)
            self.assertEqual(new_profiles, legacy_profiles)

    def test_case_2_new_generation_adds_pair_total_zero_profiles(self) -> None:
        profiles = list(iter_mtf_profiles(12, AccelerationCase.CASE_2, max_job_count=200))

        self.assertTrue(any(profile.nR2 + profile.nF2 == 0 for profile in profiles))

    def test_case_1_profiles_enforce_nf2_minus_nr3_plus_nf4_plus_nr5_plus_one_zero(self) -> None:
        profiles = [
            profile
            for profile in iter_mtf_profiles(12, AccelerationCase.CASE_1, max_job_count=200)
            if 13 in candidate_ells_for_mtf_profile(12, profile, AccelerationCase.CASE_1)
        ]

        self.assertTrue(profiles)
        for profile in profiles:
            self.assertEqual(profile.nF2 - profile.nR3 + profile.nF4 + profile.nR5 + 1, 0)

    def test_case_2_profiles_enforce_nf4_plus_nr5_plus_one_equals_nr3_minus_nf2(self) -> None:
        profiles = [
            profile
            for profile in iter_mtf_profiles(12, AccelerationCase.CASE_2, max_job_count=200)
            if 8 in candidate_ells_for_mtf_profile(12, profile, AccelerationCase.CASE_2)
        ]

        self.assertTrue(profiles)
        for profile in profiles:
            self.assertEqual(profile.nF4 + profile.nR5 + 1, profile.nR3 - profile.nF2)
            tail_total = 12 - (profile.nR2 + profile.nF2)
            self.assertEqual(
                profile.nR3 + profile.nF3 + profile.nR4 + profile.nF4 + profile.nR5,
                tail_total,
            )

    def test_case_2_fallback_starts_respect_lower_bounds_for_f2_and_f3(self) -> None:
        profile = next(
            profile
            for profile in iter_mtf_profiles(12, AccelerationCase.CASE_2, max_job_count=200)
            if profile.nF2 > 0 and profile.nF3 > 0
        )

        starts = list(iter_fallback_starts(12, profile, AccelerationCase.CASE_2))

        self.assertTrue(starts)
        prefix_total = profile.nF1 + profile.nR2 + profile.nF2
        structural_s3_min = (
            2 * prefix_total
            + profile.nF2
            + 3 * profile.nR3
            + 3 * profile.nF3
            + 2
        )
        for start in starts:
            self.assertIsNotNone(start.s2)
            self.assertIsNotNone(start.s3)
            assert start.s2 is not None
            assert start.s3 is not None
            self.assertGreaterEqual(start.s2, 2 * prefix_total + 4)
            self.assertGreaterEqual(start.s3, start.s2 + profile.nF2)
            self.assertGreaterEqual(start.s3, structural_s3_min)
            self.assertIsNone(start.s4)

    def test_case_2_fallback_starts_respect_lower_bounds_for_f4(self) -> None:
        profile = next(
            profile
            for profile in iter_mtf_profiles(11, AccelerationCase.CASE_2, max_job_count=200)
            if profile.compact_id == "mtf00030611"
        )

        starts = list(iter_fallback_starts(11, profile, AccelerationCase.CASE_2))

        self.assertTrue(starts)
        prefix_total = profile.nF1 + profile.nR2 + profile.nF2
        structural_s4_min = (
            2 * prefix_total
            + profile.nF2
            + 3 * profile.nR3
            + 4 * profile.nF3
            + 4 * profile.nR4
            + 4 * profile.nF4
            + 2
        )
        for start in starts:
            self.assertIsNone(start.s2)
            self.assertIsNotNone(start.s4)
            assert start.s4 is not None
            self.assertGreaterEqual(start.s4, structural_s4_min)

    def test_case_2_fallback_starts_exact_for_m8_n30_profile(self) -> None:
        profile = next(
            profile
            for profile in iter_mtf_profiles(8, AccelerationCase.CASE_2, max_job_count=100)
            if profile.compact_id == "mtf00140102"
        )

        starts = list(iter_fallback_starts(8, profile, AccelerationCase.CASE_2))

        self.assertEqual(
            starts,
            [
                FallbackStarts(13, None, None),
                FallbackStarts(15, None, None),
                FallbackStarts(16, None, None),
                FallbackStarts(19, None, None),
                FallbackStarts(25, None, None),
                FallbackStarts(29, None, None),
            ],
        )
        self.assertNotIn(FallbackStarts(s2=7, s3=None, s4=None), starts)
        self.assertNotIn(FallbackStarts(s2=8, s3=None, s4=None), starts)
        self.assertNotIn(FallbackStarts(s2=9, s3=None, s4=None), starts)

    def test_case_2_fallback_starts_exact_snapshot_for_multi_fallback_profile(self) -> None:
        profile = next(
            profile
            for profile in iter_mtf_profiles(12, AccelerationCase.CASE_2, max_job_count=200)
            if profile.compact_id == "mtf00361002"
        )

        starts = list(iter_fallback_starts(12, profile, AccelerationCase.CASE_2))

        self.assertEqual(len(starts), 13)
        self.assertEqual(
            starts[:10],
            [
                FallbackStarts(23, 37, None),
                FallbackStarts(23, 41, None),
                FallbackStarts(25, 37, None),
                FallbackStarts(25, 41, None),
                FallbackStarts(26, 37, None),
                FallbackStarts(26, 41, None),
                FallbackStarts(27, 37, None),
                FallbackStarts(27, 41, None),
                FallbackStarts(28, 37, None),
                FallbackStarts(28, 41, None),
            ],
        )
        self.assertEqual(
            starts[-3:],
            [
                FallbackStarts(34, 37, None),
                FallbackStarts(34, 41, None),
                FallbackStarts(38, 41, None),
            ],
        )
        self.assertNotIn(FallbackStarts(11, 36, None), starts)
        self.assertNotIn(FallbackStarts(12, 36, None), starts)
        self.assertNotIn(FallbackStarts(28, 35, None), starts)

    def test_case_2_fallback_start_cannot_create_gap_between_regular_machines(self) -> None:
        profile = MtfProfile(0, 0, 1, 2, 0, 5, 0, 0)

        starts = list(iter_fallback_starts(8, profile, AccelerationCase.CASE_2))

        self.assertIn(FallbackStarts(9, None, None), starts)
        self.assertNotIn(FallbackStarts(13, None, None), starts)

    def test_generated_profiles_match_mtf_job_count_after_reordered_branching(self) -> None:
        for acceleration_case, ell in (
            (AccelerationCase.CASE_1, 9),
            (AccelerationCase.CASE_2, 10),
        ):
            for mtf_profile in iter_mtf_profiles(8, acceleration_case, max_job_count=100):
                if ell not in candidate_ells_for_mtf_profile(8, mtf_profile, acceleration_case):
                    continue
                for opt_profile in iter_opt_profiles(
                    8,
                    ell,
                    acceleration_case,
                    mtf_profile=mtf_profile,
                ):
                    self.assertEqual(mtf_profile.total_job_count, opt_profile.total_job_count)


if __name__ == "__main__":
    unittest.main()
