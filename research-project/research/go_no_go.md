# GO

Debt Collection → Issue feasibility pilot, 2 October 2026.

The fixed numerical rule passes **4 of 4** conditions: **GO**. Special warning checks: none triggered under the prespecified operational definitions.

| Condition | Result | Evidence |
|---|---|---|
| 1. Non-trivial task | PASS | Future accuracy 47.4%–56.0%; macro-F1 0.349–0.394. |
| 2. Sufficient future data | PASS | Minimum 281 distinct families per class/primary quarter; 34,343 exact-deduplicated training rows. |
| 3. Temporal/confidence behavior | PASS | Useful confidence separation in 8 windows; future accuracy range 8.6%, ECE range 0.046. |
| 4. Manageable leakage | PASS | Historical family exposure at most 13.8%; minimum 281 novel families per class/primary quarter. Narrative-only tests passed. |

The thresholds interpreting qualitative conditions were recorded in `configs/data.yaml` before classifier fitting. At least 200 families per class/quarter is a pilot adequacy threshold, not a claim that subgroup gate accuracy can be estimated precisely: at 90% correctness, 200 independent examples have a rough 95% margin of ±4.2 percentage points, before selective acceptance reduces the sample.

A family is a lexical dependence proxy, not verified consumer identity. Results apply to published narratives. Label ambiguity, within-period templates and changing publication rates remain important limitations even when the leakage condition passes.

This result does not establish a Laya advantage, paper novelty or a safe operational 90% gate. No transformer, neural model or GPU experiment was run. STOP: explicit user approval is required for any larger experiment.
