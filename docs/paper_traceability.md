# Paper Traceability

This document is the working paper-to-code traceability table for the current
`multifit-optveri` implementation.

The immediate goal is not to prove the paper correct. The goal is narrower and
more operational:

1. identify each mathematical claim used by the implementation,
2. point to the exact code location responsible for it,
3. classify whether the implementation is exact, stronger, weaker, or simply
   encoded in a different form.

## Status Legend

- `EXACT`: same mathematical meaning as the paper statement.
- `DIFFERENT FORM`: same intended meaning, but encoded differently
  (for example as a variable bound instead of an explicit constraint).
- `STRONGER`: the code enforces the paper statement and additional restriction.
- `WEAKER`: the code relaxes the paper statement in some way.
- `MISSING`: not yet implemented.
- `UNCLEAR`: implemented, but the exact paper correspondence still needs
  a careful proof or review.

## Code Map

- `src/multifit_optveri/models/obv.py`
  Section 4 base MIQP, global strengthenings, Section 5 common cuts, and
  case-specific constraints.
- `src/multifit_optveri/branching.py`
  Outer branching logic: `ell_iterator`, `MTF_profiles`, `OPT_profiles`.
- `src/multifit_optveri/experiments.py`
  Materializes the paper-style outer loops into concrete experiment cases.
- `src/multifit_optveri/acceleration.py`
  Top-level `p_n` case partition and paper regime guards.

## Section 4 Base MIQP

| Paper Ref | Paper Statement | Code Location | Implementation Form | Status | Note |
|---|---|---|---|---|---|
| Sec. 4 vars | decision variables `p, x, q, s, Z` | `build_obv_model` in `models/obv.py` | direct variable creation | EXACT | variable names are `p`, `x`, `q`, `s`, `z_var` |
| Sec. 4 sorting | `p_j >= p_{j+1}` | `build_obv_model` / constraint name `sorting` | explicit linear constraints | EXACT | uses `truncated_jobs = {1,...,n-1}` |
| Sec. 4 OPT assignment | each job assigned once in OPT | `build_obv_model` / `opt_assign` | explicit linear constraints | EXACT | |
| Sec. 4 OPT makespan | load of each OPT machine at most 1 | `build_obv_model` / `opt_makespan` | explicit linear constraints | EXACT | |
| Sec. 4 MTF assignment | each job in `{1,...,n-1}` assigned once in MTF | `build_obv_model` / `mtf_assign` | explicit linear constraints | EXACT | |
| Sec. 4 MTF init | `s_{i1} = q_{i1} p_1` | `build_obv_model` / `mtf_init` | explicit bilinear constraints | EXACT | |
| Sec. 4 MTF load update | `s_{ij} - s_{i,j-1} = q_{ij} p_j` | `build_obv_model` / `mtf_contribution` | explicit bilinear constraints | EXACT | |
| Sec. 4 MTF feasibility | `s_{ij} <= rho` | `build_obv_model` / `mtf_feasible` | explicit linear constraints | EXACT | |
| Sec. 4 MTF logic | if job `j` is not assigned up to machine `i`, machine `i` was already too full | `build_obv_model` / `mtf_logic` | explicit linear constraints | EXACT | encoded in standard big-M style used in the current model |
| Sec. 4 objective lower envelope | `s_{i,n-1} + p_n >= Z` | `build_obv_model` / `mtf_objective` | explicit linear constraints | EXACT | |
| verification target | only care about counterexamples with objective at least target | `build_obv_model` / `target_lb` | explicit linear constraint | EXACT | this is a verification-mode restriction, not part of the bare existence formulation |

## General Valid Inequalities And Initialization

### Table `valid_init`

| Paper Ref | Paper Statement | Code Location | Implementation Form | Status | Note |
|---|---|---|---|---|---|
| Obs. `p_n` | `p_n > m(\rho-1)/(m-1)` | `_processing_time_lower_bound`, `build_obv_model` in `models/obv.py` | variable lower bound on every `p_j` | DIFFERENT FORM | same lower-bound idea, but implemented as a bound on all `p_j`; strict `>` becomes non-strict bound in practice |
| Obs. `opt3` bound | `p_j <= min(1/(1+floor((j-1)/m)), 1 - 2m(\rho-1)/(m-1))` | `_processing_time_upper_bound`, `build_obv_model` in `models/obv.py` | variable upper bound on every `p_j` | DIFFERENT FORM | same idea, encoded as variable bound rather than `addConstr` |
| Obs. `opt3` cardinality lb | `|OPT_i| >= 3` | `_apply_global_valid_inequalities` / `opt_cardinality_lb` in `models/obv.py` | explicit linear constraints | EXACT | |
| Sym. Brk. MTF prefix | `q_{ij}=0` for `i>j` | `_apply_global_valid_inequalities` / `mtf_init_order` in `models/obv.py` | explicit linear constraints | EXACT | |
| Sym. Brk. MTF init | `s_{11}=p_1`, `s_{i1}=0` for `i>1` | `_apply_global_valid_inequalities` / `mtf_init_fixed[1]`, `mtf_init_fixed` in `models/obv.py` | explicit linear constraints | EXACT | |
| Sym. Brk. OPT cardinality order | `|OPT_i| <= |OPT_{i+1}|` | `_apply_global_valid_inequalities` / `opt_cardinality_order` in `models/obv.py` | explicit linear constraints | EXACT | this now applies in base and accelerated runs |
| Fact `valid-ineq` MTF ub | `|MTF_i| <= ceil((m-\rho)/(m(\rho-1))) - 1` | `_apply_global_valid_inequalities` / `mtf_cardinality_ub` in `models/obv.py` | explicit linear constraints | EXACT | |
| Fact `valid-ineq` OPT ub | `|OPT_i| <= ceil((m-1)/(m(\rho-1))) - 1` | `_apply_global_valid_inequalities` / `opt_cardinality_ub` in `models/obv.py` | explicit linear constraints | EXACT | |
| Additional tail-sum valid inequality | `sum_{j=n-4|S4|-5|S5|+1}^n p_j <= |S4|+|S5|` | `_apply_opt_profile_tail_sum_constraint` in `models/obv.py` | explicit linear constraint | DIFFERENT FORM | implemented once an `opt_profile` is fixed, not as a profile-free global cut |

### Additional General Constraints Present In Code

These are not listed verbatim in the table above, but are present in the code.

| Code Constraint | Code Location | Status | Note |
|---|---|---|---|
| `opt_smallest_jobs_sum` | `_apply_global_valid_inequalities` in `models/obv.py` | UNCLEAR | useful strengthening; not currently mapped to a single line in the paper tables |
| `mtf_cardinality_lb` | `_apply_global_valid_inequalities` in `models/obv.py` | UNCLEAR | stronger initialization/valid inequality used by the implementation |
| `mtf_balance` | `_apply_global_valid_inequalities` in `models/obv.py` | UNCLEAR | derived/redundant balancing constraint retained in the implementation |

## General Profile-Based Conditions Before Case Branching

### `D / D'` Split and Profile Layout

| Paper Ref | Paper Statement | Code Location | Implementation Form | Status | Note |
|---|---|---|---|---|---|
| `JobSetLong` | `p_j > 3/17 + p_n` for long jobs | `_apply_profile_cardinality_constraints` in `models/obv.py` | explicit linear constraints on jobs `< ell` | WEAKER | strict `>` encoded as `>=` |
| `JobSetShort` | `p_j <= 3/17 + p_n` for short jobs | `_apply_profile_cardinality_constraints` in `models/obv.py` | explicit linear constraints on jobs `>= ell` | EXACT | |
| general index layout | definitions of `e2, e3, e4, t2, t3, t4` | `_build_mtf_profile_layout` in `models/obv.py` | explicit arithmetic on profile counts | EXACT | compare this helper directly with the table text |

### Table `cond_constr_general`

| Paper Ref | Paper Statement | Code Location | Implementation Form | Status | Note |
|---|---|---|---|---|---|
| `F1` cardinality | `|MTF_i| = 2` for `i in F1` | `mtf_profile_cardinality[...]` in `_apply_profile_cardinality_constraints` | exact profile cardinality fixing | EXACT | imposed via chosen `mtf_profile` |
| `q_{ii}=1` for `i in [e2]` / diagonal F1 prefix | `_apply_mtf_base_profile_constraints` / `F1_assignment_constr[...]` | explicit linear constraints | EXACT | together with later prefix cases this covers the diagonal placement behavior |
| `F1` valid inequality | earliest pair on `F1[-1]` already blocks `p_n` | `_apply_mtf_base_profile_constraints` / `F1_valid_constr[...]` | explicit linear constraint | EXACT | |
| `R2` cardinality | `|MTF_i| = 2` for `i in R2` | `mtf_profile_cardinality[...]` in `_apply_profile_cardinality_constraints` | exact profile cardinality fixing | EXACT | |
| `R2` valid inequality | `p_{e2+2|R2|-2} + p_{e2+2|R2|-1} + p_n >= 20/17` | `_apply_mtf_base_profile_constraints` / `R2_valid_constr[...]` | explicit linear constraint | EXACT | |
| `R2` equal-processing | `p_j = p_{j+1}` for `j in [t2+1, e2+2|R2|-2)` | `_apply_mtf_base_profile_constraints` / `R2_processing_times[...]` | explicit equalities | EXACT | recently aligned to start at `t2 + 1` |
| `R2` OPT lex order | `sum_{i'=1}^i x_{i'j} >= x_{i,j+1}` over same range | `_apply_mtf_base_profile_constraints` / `R2_symmetry_break_by_proc[...]` | explicit inequalities | EXACT | |
| `F2` cardinality | `|MTF_i| = 3` for `i in F2` | `mtf_profile_cardinality[...]` in `_apply_profile_cardinality_constraints` | exact profile cardinality fixing | EXACT | |
| `F2` valid inequality | `sum_{k=0}^2 p_{e3-k} >= 20/17` | `_apply_mtf_base_profile_constraints` / `F2_valid_constr[...]` | explicit linear constraint | EXACT | |
| `R3` cardinality | `|MTF_i| = 3` for `i in R3` | `mtf_profile_cardinality[...]` in `_apply_profile_cardinality_constraints` | exact profile cardinality fixing | EXACT | |
| `R3` valid inequality | earliest job on `R3[-1]` plus `p_n` blocks | `_apply_mtf_base_profile_constraints` / `R3_valid_constr[...]` | explicit linear constraint | EXACT | |
| `R3` equal-processing | `p_j = p_{j+1}` for `j in [t3+1, e3+3|R3|-3)` | `_apply_mtf_base_profile_constraints` / `R3_processing_times[...]` | explicit equalities | EXACT | |
| `R3` OPT lex order | same lex order over the `R3` equal-processing range | `_apply_mtf_base_profile_constraints` / `R3_symmetry_break_by_proc[...]` | explicit inequalities | EXACT | |
| `F3` cardinality | `|MTF_i| = 4` for `i in F3` | `mtf_profile_cardinality[...]` in `_apply_profile_cardinality_constraints` | exact profile cardinality fixing | EXACT | |
| `F3` valid inequality | `sum_{k=0}^3 p_{e4-k} >= 20/17` | `_apply_mtf_base_profile_constraints` / `F3_valid_constr[...]` | explicit linear constraint | EXACT | |
| `R4` cardinality | `|MTF_i| = 4` for `i in R4` | `mtf_profile_cardinality[...]` in `_apply_profile_cardinality_constraints` | exact profile cardinality fixing | EXACT | |
| `R4` valid inequality | earliest job on `R4[-1]` plus `p_n` blocks | `_apply_mtf_base_profile_constraints` / `R4_valid_constr[...]` | explicit linear constraint | EXACT | |
| `R4` equal-processing | `p_j = p_{j+1}` for `j in [t4+1, e4+4|R4|-4)` | `_apply_mtf_base_profile_constraints` / `R4_processing_times[...]` | explicit equalities | EXACT | |
| `R4` OPT lex order | same lex order over the `R4` equal-processing range | `_apply_mtf_base_profile_constraints` / `R4_symmetry_break_by_proc[...]` | explicit inequalities | EXACT | |
| `M5` cardinality | `|MTF_i| = 5` for `i in M5` | `_apply_mtf_base_profile_constraints` / `M5_cardinality_constr[...]` | explicit linear constraints | EXACT | |
| `R5` equal-processing analogue | lemma-driven equal-processing for regular 5-machine interiors | `_apply_r5_constraints` / `case34_M5_processing_times[...]` in `models/obv.py` | explicit equalities | EXACT | only appears when the case-specific branch reaches the `M5` region |

## Section 5 Common Case Partition

| Paper Ref | Paper Statement | Code Location | Implementation Form | Status | Note |
|---|---|---|---|---|---|
| top-level partition | `case_1: p_n >= 11/51`, `case_2: 7/34 <= p_n < 11/51`, `case_3: p_n < 7/34` | `AccelerationCase.pn_range` in `acceleration.py` | enum-backed interval definition | EXACT | strict upper bounds are later encoded non-strictly in the model |
| paper common lower bound | `p_n > 3m/(17(m-1))` for `8 <= m <= 12` | `paper_common_pn_lower_bound` in `acceleration.py` | helper function | EXACT | |
| common Section 5 cuts | common `p_n` bound and per-machine size caps before case specifics | `_apply_paper_acceleration_constraints` in `models/obv.py` | explicit constraints | MOSTLY EXACT | the `p_n` upper bounds are encoded as `<=` rather than strict `<` |

## Case 1

| Paper Ref | Paper Statement | Code Location | Implementation Form | Status | Note |
|---|---|---|---|---|---|
| outer algorithm | `ell_iterator(m, case_1)` | `ell_iterator` in `branching.py` | descending iterator over odd `ell` values | EXACT | current paper interpretation is `ell` odd up to `m+1` |
| outer algorithm | `OPT_profiles(m, ell, n, case_1)` | `iter_opt_profiles` in `branching.py` | single OPT profile `(ell-1, m-ell+1, 0)` | EXACT | |
| outer algorithm | `MTF_profiles(m, ell, case_1)` | `iter_mtf_profiles` in `branching.py` | profile enumeration with direct `n = 4m - ell + 1` filtering | EXACT | now enforces the Case 1 total job-count identity directly |
| outer algorithm | `n <- n(pi')` | `enumerate_cases` in `experiments.py` | `job_count = mtf_profile.total_job_count` | EXACT | |
| Case 1 stronger `D'` bound | `p_j < 2(7/17 - p_n)` for short jobs | `_apply_case_profile_constraints` / `case1_stronger_proc_time_in_D` in `models/obv.py` | explicit inequality | WEAKER | encoded as `<=` |
| Case 1 OPT profile | long-job machines are 3-job OPT machines, rest are 4-job machines | `iter_opt_profiles` + `opt_profile_cardinality[...]` | profile generation plus fixed cardinalities | EXACT | |
| Case 1 OPT diagonal anchor | `x_{ii}=1` for long-job machines | `_apply_case_profile_constraints` / `case1_OPT_ell_assignment_constr` | explicit equalities | EXACT | |
| Case 1 OPT init symmetry | `x_{ij}=0` for `i>j` | `_apply_case_profile_constraints` / `case1_opt_init` | explicit inequalities | EXACT | |
| Case 1 no `F1` | `|F1| = 0` | `iter_mtf_profiles` in `branching.py` | profile tuple always has first component 0 | EXACT | encoded in the branch object rather than a model constraint |
| Case 1 pair block identity | `2(|R2|+|F2|)+1 = ell` | `iter_mtf_profiles` in `branching.py` | pair-count arithmetic | EXACT | |
| Case 1 total job identity | `2|R2| + 3(|F2|+|R3|) + 4(|F3|+|R4|) + 5|M5| = 4m - ell + 1` | `iter_mtf_profiles` in `branching.py` | direct total-job-count filter | EXACT | |
| Case 1 `R2` adjacency | `{2i-1, 2i} subseteq JobinMTF_i` for `i in R2` | `_apply_case_profile_constraints` / `case1_R2_consec_*` | explicit equalities | EXACT | |
| Case 1 `F2` adjacency | `{2i-1, 2i} subseteq JobinMTF_i` for `i in F2` | `_apply_case_profile_constraints` / `case1_F2_consec_*` | explicit equalities | EXACT | |
| Case 1 `F2` valid inequality | `p_{ell-2}+p_{ell-1}+p_{ell+2} >= 20/17` | `_apply_case_profile_constraints` / `case1_F2_valid_constr` | explicit inequality | EXACT | |
| Case 1 first `R3` anchor | `R3[1] = {ell, ell+1, ell+2}` | `_apply_case_profile_constraints` / `case1_R3_consec_*` | explicit equalities | EXACT | |

## Next Work Items

These are the most important remaining traceability gaps after the first pass.

- extend this table through Case 2, Case 3-1, and Case 3-2;
- classify which old-code strengthenings are truly paper-backed and which are
  purely implementation-side speedups;
- add exact LP-level spot checks for one representative branch in each case.
