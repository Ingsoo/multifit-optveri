from __future__ import annotations

from fractions import Fraction
import unittest

from multifit_optveri.math_utils import (
    ceil_fraction,
    format_decimal_number,
    format_pretty_number,
    format_ratio,
    format_scaled_rational_values,
    format_sorted_decimal_values,
    format_sorted_numeric_values,
    parse_ratio,
)


class MathUtilsTests(unittest.TestCase):
    def test_parse_ratio_accepts_multiple_input_types(self) -> None:
        self.assertEqual(parse_ratio(Fraction(3, 5)), Fraction(3, 5))
        self.assertEqual(parse_ratio(7), Fraction(7, 1))
        self.assertEqual(parse_ratio(0.25), Fraction(1, 4))
        self.assertEqual(parse_ratio("20/17"), Fraction(20, 17))
        self.assertEqual(parse_ratio("9"), Fraction(9, 1))

    def test_ceil_fraction_rounds_up(self) -> None:
        self.assertEqual(ceil_fraction(Fraction(7, 3)), 3)
        self.assertEqual(ceil_fraction(Fraction(9, 3)), 3)
        self.assertEqual(ceil_fraction(Fraction(1, 5)), 1)

    def test_format_ratio_renders_integer_and_fraction(self) -> None:
        self.assertEqual(format_ratio(Fraction(12, 1)), "12")
        self.assertEqual(format_ratio(Fraction(20, 17)), "20/17")

    def test_format_pretty_number_prefers_clean_fractions(self) -> None:
        self.assertEqual(format_pretty_number(0.3333333333333), "1/3")
        self.assertEqual(format_pretty_number(0.4999999999999), "1/2")
        self.assertEqual(format_pretty_number(-1e-12), "0")

    def test_format_sorted_numeric_values_sorts_and_formats(self) -> None:
        rendered = format_sorted_numeric_values([0.2, 0.333333333333, 0.199999999999, 0.5])
        self.assertEqual(rendered, "[1/2, 1/3, 1/5, 1/5]")

    def test_format_scaled_rational_values_uses_common_denominator(self) -> None:
        rendered = format_scaled_rational_values(
            [0.529411710413, 0.411764744663, 0.35294113769, 0.294117658693, 0.235294114063]
        )
        self.assertEqual(rendered, "[9, 7, 6, 5, 4]")

    def test_format_scaled_rational_values_handles_mixed_exact_and_approximate_inputs(self) -> None:
        rendered = format_scaled_rational_values(
            [
                0.529411073233,
                0.529411073233,
                0.529411073233,
                0.411765397355,
                float(Fraction(6, 17)),
                0.294117992795,
                0.294117301322,
                0.235294232893,
                0.235294232893,
                float(Fraction(4, 17)),
            ]
        )
        self.assertEqual(rendered, "[9, 9, 9, 7, 6, 5, 5, 4, 4, 4]")

    def test_decimal_helpers_render_plain_decimal_lists(self) -> None:
        self.assertEqual(format_decimal_number(0.500000000001), "0.500000000001")
        self.assertEqual(
            format_sorted_decimal_values([0.2, 0.3333333, 0.1999999]),
            "[0.3333333, 0.2, 0.1999999]",
        )


if __name__ == "__main__":
    unittest.main()
