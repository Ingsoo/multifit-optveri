from __future__ import annotations

import unittest

from multifit_optveri.acceleration import AccelerationCase
from multifit_optveri.models.spec import derive_obv_dimensions


class ModelSpecTests(unittest.TestCase):
    def test_dimension_counts_for_small_case(self) -> None:
        dimensions = derive_obv_dimensions(8, 24, include_target_lower_bound=True)

        self.assertEqual(dimensions.variable_counts["p"], 24)
        self.assertEqual(dimensions.variable_counts["x"], 192)
        self.assertEqual(dimensions.variable_counts["r"], 192)
        self.assertEqual(dimensions.variable_counts["q"], 184)
        self.assertEqual(dimensions.variable_counts["s"], 184)
        self.assertEqual(dimensions.variable_counts["w"], 184)
        self.assertEqual(dimensions.total_variables, 961)
        self.assertEqual(dimensions.constraint_counts["opt_cardinality_lb"], 8)
        self.assertEqual(dimensions.constraint_counts["opt_cardinality_ub"], 8)
        self.assertEqual(dimensions.constraint_counts["opt_cardinality_order"], 7)
        self.assertEqual(dimensions.constraint_counts["mtf_init_order"], 28)
        self.assertEqual(dimensions.constraint_counts["mtf_init_fixed"], 8)
        self.assertEqual(dimensions.constraint_counts["mtf_cardinality_lb"], 8)
        self.assertEqual(dimensions.constraint_counts["mtf_cardinality_ub"], 8)
        self.assertEqual(dimensions.constraint_counts["mtf_balance"], 22)
        self.assertEqual(dimensions.total_constraints, 2232)

    def test_dimension_counts_for_case_2_acceleration(self) -> None:
        dimensions = derive_obv_dimensions(
            8,
            24,
            include_target_lower_bound=True,
            acceleration_case=AccelerationCase.CASE_2,
        )

        self.assertEqual(dimensions.constraint_counts["pn_common_lb"], 1)
        self.assertEqual(dimensions.constraint_counts["opt_cardinality"], 8)
        self.assertEqual(dimensions.constraint_counts["opt_cardinality_order"], 7)
        self.assertEqual(dimensions.constraint_counts["mtf_cardinality"], 8)
        self.assertEqual(dimensions.constraint_counts["case_pn_lb"], 1)
        self.assertEqual(dimensions.constraint_counts["case_pn_ub"], 1)
        self.assertEqual(dimensions.total_constraints, 2251)

    def test_dimension_counts_with_profile_cardinality_constraints(self) -> None:
        dimensions = derive_obv_dimensions(
            8,
            24,
            include_target_lower_bound=True,
            acceleration_case=AccelerationCase.CASE_2,
            include_profile_cardinality_constraints=True,
        )

        self.assertEqual(dimensions.constraint_counts["opt_profile_cardinality"], 8)
        self.assertEqual(dimensions.constraint_counts["mtf_profile_cardinality"], 8)
        self.assertEqual(dimensions.total_constraints, 2267)


if __name__ == "__main__":
    unittest.main()
