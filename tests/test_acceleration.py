from __future__ import annotations

from fractions import Fraction
import unittest

from multifit_optveri.acceleration import (
    AccelerationCase,
    PnRange,
    paper_common_pn_lower_bound,
    parse_acceleration_case,
)


class AccelerationTests(unittest.TestCase):
    def test_pn_range_text_variants(self) -> None:
        self.assertEqual(PnRange().text, "unrestricted")
        self.assertEqual(PnRange(lower=Fraction(11, 51)).text, "p_n >= 11/51")
        self.assertEqual(PnRange(upper=Fraction(7, 34)).text, "p_n <= 7/34")
        self.assertEqual(
            PnRange(lower=Fraction(7, 34), upper=Fraction(11, 51)).text,
            "7/34 <= p_n <= 11/51",
        )

    def test_parse_acceleration_case_normalizes_hyphenated_input(self) -> None:
        self.assertIs(parse_acceleration_case("case-3-2"), AccelerationCase.CASE_3_2)
        self.assertIs(parse_acceleration_case(AccelerationCase.CASE_1), AccelerationCase.CASE_1)

    def test_parse_acceleration_case_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValueError):
            parse_acceleration_case("case_9")

    def test_acceleration_case_ranges_and_flags(self) -> None:
        self.assertEqual(AccelerationCase.BASE.pn_range, PnRange())
        self.assertEqual(
            AccelerationCase.CASE_1.pn_range,
            PnRange(lower=Fraction(11, 51)),
        )
        self.assertEqual(
            AccelerationCase.CASE_2.pn_range,
            PnRange(lower=Fraction(7, 34), upper=Fraction(11, 51)),
        )
        self.assertEqual(
            AccelerationCase.CASE_3_1.pn_range,
            PnRange(upper=Fraction(7, 34)),
        )
        self.assertFalse(AccelerationCase.BASE.uses_paper_acceleration)
        self.assertTrue(AccelerationCase.CASE_3_2.uses_paper_acceleration)

    def test_common_pn_lower_bound_matches_paper_formula(self) -> None:
        self.assertEqual(paper_common_pn_lower_bound(8), Fraction(24, 119))
        self.assertEqual(paper_common_pn_lower_bound(12), Fraction(36, 187))
        with self.assertRaises(ValueError):
            paper_common_pn_lower_bound(7)


if __name__ == "__main__":
    unittest.main()
