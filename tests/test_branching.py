from __future__ import annotations

import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.branching import MtfProfile, OptProfile, ell_iterator, iter_mtf_profiles, iter_opt_profiles


class BranchingTests(unittest.TestCase):
    def test_mtf_profile_properties(self) -> None:
        profile = MtfProfile(1, 2, 1, 1, 1, 1, 1)

        self.assertEqual(profile.machine_count, 8)
        self.assertEqual(profile.scheduled_job_count, 25)
        self.assertEqual(profile.total_job_count, 26)
        self.assertEqual(profile.compact_id, "mtf1211111")
        self.assertEqual(profile.machine_cardinalities, (2, 2, 2, 3, 3, 4, 4, 5))
        self.assertEqual(profile.nF1, 1)
        self.assertEqual(profile.nR4, 1)
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
        self.assertEqual(ell_iterator(8, AccelerationCase.CASE_3_2, max_job_count=100), tuple(range(12, 3, -1)))

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
        case_1_profiles = list(
            iter_mtf_profiles(
                8,
                9,
                OptProfile(8, 0, 0, pattern="case1"),
                AccelerationCase.CASE_1,
            )
        )
        case_2_profiles = list(
            iter_mtf_profiles(
                8,
                10,
                OptProfile(8, 0, 0, pattern="two_long"),
                AccelerationCase.CASE_2,
            )
        )

        self.assertEqual(len(case_1_profiles), 13)
        self.assertEqual(
            [profile.compact_id for profile in case_1_profiles[:3]],
            ["mtf0402011", "mtf0401030", "mtf0402101"],
        )
        self.assertEqual(len(case_2_profiles), 3)
        self.assertEqual(
            [profile.compact_id for profile in case_2_profiles],
            ["mtf0401030", "mtf0312020", "mtf0223010"],
        )

    def test_case_3_profiles_satisfy_basic_invariants(self) -> None:
        opt_profile = OptProfile(0, 8, 0, pattern="generic")
        case_31_profiles = list(
            iter_mtf_profiles(8, 4, opt_profile, AccelerationCase.CASE_3_1)
        )
        case_32_profiles = list(
            iter_mtf_profiles(
                8,
                4,
                OptProfile(0, 1, 7, pattern="generic"),
                AccelerationCase.CASE_3_2,
            )
        )

        self.assertEqual(
            [profile.compact_id for profile in case_31_profiles],
            [
                "mtf0001070",
                "mtf0002051",
                "mtf0003032",
                "mtf0004013",
                "mtf0010070",
                "mtf0011051",
                "mtf0012032",
                "mtf0013013",
            ],
        )
        self.assertEqual([profile.compact_id for profile in case_32_profiles], ["mtf0010007"])

        for profile in case_31_profiles + case_32_profiles:
            self.assertEqual(profile.machine_count, 8)

    def test_generated_profiles_match_opt_job_count(self) -> None:
        for acceleration_case, ell, opt_profile in (
            (AccelerationCase.CASE_1, 9, OptProfile(8, 0, 0, pattern="case1")),
            (AccelerationCase.CASE_2, 10, OptProfile(8, 0, 0, pattern="two_long")),
        ):
            for mtf_profile in iter_mtf_profiles(8, ell, opt_profile, acceleration_case):
                self.assertEqual(mtf_profile.total_job_count, opt_profile.total_job_count)


if __name__ == "__main__":
    unittest.main()
