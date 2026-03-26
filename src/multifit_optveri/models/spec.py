from __future__ import annotations

from dataclasses import dataclass

from multifit_optveri.acceleration import AccelerationCase

# This file is only a coarse dimension/spec counter for the generated model.
# It is useful for sanity-checking Section 4 counts, but it is NOT a full proof
# that the implemented model equals the paper: many case-specific structural
# constraints in `models/obv.py` are more detailed than the counts tracked here.

@dataclass(frozen=True)
class ObvModelDimensions:
    machine_count: int
    job_count: int
    acceleration_case: AccelerationCase
    variable_counts: dict[str, int]
    constraint_counts: dict[str, int]
    include_target_lower_bound: bool

    @property
    def total_variables(self) -> int:
        return sum(self.variable_counts.values())

    @property
    def total_constraints(self) -> int:
        return sum(self.constraint_counts.values())


def derive_obv_dimensions(
    machine_count: int,
    job_count: int,
    *,
    include_target_lower_bound: bool,
    acceleration_case: AccelerationCase = AccelerationCase.BASE,
    include_profile_cardinality_constraints: bool = False,
) -> ObvModelDimensions:
    # Use this when you want to compare "how many" major families of variables
    # and constraints exist. For exact paper/code alignment, you still need to
    # inspect the named constraints added in `models/obv.py`.
    if machine_count <= 0:
        raise ValueError("machine_count must be positive.")
    if job_count <= 1:
        raise ValueError("job_count must be at least 2.")

    variable_counts = {
        "p": job_count,
        "x": machine_count * job_count,
        "q": machine_count * (job_count - 1),
        "s": machine_count * (job_count - 1),
        "Z": 1,
    }

    mtf_init_order_count = sum(max(machine_count - job_index, 0) for job_index in range(1, job_count))

    constraint_counts = {
        "sorting": job_count - 1,
        "opt_assign": job_count,
        "opt_makespan": machine_count,
        "mtf_assign": job_count - 1,
        "mtf_init": machine_count,
        "mtf_contribution": machine_count * (job_count - 2),
        "mtf_feasible": machine_count * (job_count - 1),
        "mtf_logic": machine_count * (job_count - 2),
        "mtf_objective": machine_count,
        "opt_cardinality_lb": machine_count,
        "opt_cardinality_ub": machine_count,
        "mtf_init_order": mtf_init_order_count,
        "mtf_init_fixed": machine_count,
        "mtf_cardinality_lb": machine_count,
        "mtf_cardinality_ub": machine_count,
        "mtf_balance": job_count - 2,
    }

    if include_target_lower_bound:
        constraint_counts["target_lb"] = 1

    if job_count % machine_count != 0:
        constraint_counts["opt_smallest_jobs_sum"] = 1

    if acceleration_case is not AccelerationCase.BASE:
        constraint_counts["pn_common_lb"] = 1
        constraint_counts["opt_cardinality"] = machine_count
        constraint_counts["opt_cardinality_order"] = machine_count - 1
        constraint_counts["mtf_cardinality"] = machine_count

        pn_range = acceleration_case.pn_range
        if pn_range.lower is not None:
            constraint_counts["case_pn_lb"] = 1
        if pn_range.upper is not None:
            constraint_counts["case_pn_ub"] = 1

    if include_profile_cardinality_constraints:
        constraint_counts["opt_profile_cardinality"] = machine_count
        constraint_counts["mtf_profile_cardinality"] = machine_count

    return ObvModelDimensions(
        machine_count=machine_count,
        job_count=job_count,
        acceleration_case=acceleration_case,
        variable_counts=variable_counts,
        constraint_counts=constraint_counts,
        include_target_lower_bound=include_target_lower_bound,
    )
