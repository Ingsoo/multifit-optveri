from __future__ import annotations

import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import MtfProfile, OptProfile, ell_iterator, iter_mtf_profiles, iter_opt_profiles


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
        self.assertEqual(profile.nM5, 1)

    def test_opt_profile_properties(self) -> None:
        profile = OptProfile(3, 4, 1, pattern="two_long")

        self.assertEqual(profile.machine_count, 8)
        self.assertEqual(profile.total_job_count, 30)
        self.assertEqual(profile.compact_id, "opt341twolong")
        self.assertEqual(profile.machine_cardinalities, (3, 3, 3, 4, 4, 4, 4, 5))
        self.assertEqual(profile.nS3, 3)
        self.assertEqual(profile.nS5, 1)

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
        self.assertEqual(
            list(iter_opt_profiles(8, 10, AccelerationCase.CASE_2)),
            [OptProfile(8, 0, 0, pattern="two_long")],
        )
        self.assertEqual(
            list(iter_opt_profiles(8, 9, AccelerationCase.CASE_2)),
            [OptProfile(8, 0, 0, pattern="regular")],
        )

    def test_case_1_and_case_2_mtf_profile_counts_match_current_branching(self) -> None:
        raw_case_1_profiles = list(
            iter_mtf_profiles(
                8,
                9,
                AccelerationCase.CASE_1,
                max_job_count=100,
            )
        )
        case_1_profiles = list(
            profile
            for profile in raw_case_1_profiles
            if any(
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
            for profile in iter_mtf_profiles(
                8,
                10,
                AccelerationCase.CASE_2,
                max_job_count=100,
            )
            if any(
                candidate == OptProfile(8, 0, 0, pattern="two_long")
                for candidate in iter_opt_profiles(
                    8,
                    10,
                    AccelerationCase.CASE_2,
                    mtf_profile=profile,
                )
            )
        )

        self.assertTrue(all(profile.total_job_count == 24 for profile in raw_case_1_profiles))
        self.assertEqual(len(case_1_profiles), 16)
        self.assertEqual(
            [profile.compact_id for profile in case_1_profiles[:3]],
            ["mtf04020101", "mtf04020110", "mtf04010300"],
        )
        self.assertEqual(len(case_2_profiles), 4)
        self.assertEqual(
            [profile.compact_id for profile in case_2_profiles],
            ["mtf04010300", "mtf03120200", "mtf02230100", "mtf01340000"],
        )

    def test_case_3_profiles_satisfy_basic_invariants(self) -> None:
        case_31_profiles = list(
            profile
            for profile in iter_mtf_profiles(
                8,
                4,
                AccelerationCase.CASE_3_1,
                max_job_count=32,
            )
            if any(
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
            for profile in iter_mtf_profiles(
                8,
                4,
                AccelerationCase.CASE_3_2,
                max_job_count=39,
            )
            if any(
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
                "mtf00010700",
                "mtf00020501",
                "mtf00030302",
                "mtf00040103",
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
        profiles = list(
            iter_mtf_profiles(
                12,
                8,
                AccelerationCase.CASE_2,
                max_job_count=200,
            )
        )

        self.assertTrue(any(profile.nF4 > 0 for profile in profiles))

    def test_generated_profiles_match_mtf_job_count_after_reordered_branching(self) -> None:
        for acceleration_case, ell in (
            (AccelerationCase.CASE_1, 9),
            (AccelerationCase.CASE_2, 10),
        ):
            for mtf_profile in iter_mtf_profiles(
                8,
                ell,
                acceleration_case,
                max_job_count=100,
            ):
                for opt_profile in iter_opt_profiles(
                    8,
                    ell,
                    acceleration_case,
                    mtf_profile=mtf_profile,
                ):
                    self.assertEqual(mtf_profile.total_job_count, opt_profile.total_job_count)


if __name__ == "__main__":
    unittest.main()
