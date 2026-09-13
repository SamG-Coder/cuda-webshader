# Chrono input-to-WGSL optimization review

Scope: the complete active Chrono pipeline (15 CUDA entry points), its emitted reachable WGSL functions, and the shared sort/scan runtime used between those kernels. This is not a review of every unrelated showcase in the repository. Work remains on the experimental branch; no production compiler, CUDA source, timestep, or neighbour-rebuild policy was changed.

## Measurement before optimization

`profile-chrono-generated.mjs` records a per-function generated-code inventory and hardware timestamps. To attribute work, it separates dispatches into timestamped compute passes. Consequently its timings are diagnostic, affected by timestamp quantization and changed pass boundaries, and must not be used as production wall-time benchmarks. Copies, readbacks, CPU work and waits are excluded.

In the 50-step baseline, the measured compute total was 70.386 ms:

| Work | Timestamp total | Dispatches |
| --- | ---: | ---: |
| RHS force derivatives | 38.207 ms | 100 |
| Stable bitonic sort stages | 11.207 ms | 6,120 |
| Neighbour-list fill | 6.750 ms | 51 |
| Neighbour counts | 4.915 ms | 51 |
| Adami boundary update | 3.146 ms | 100 |
| XSPH shifting | 2.818 ms | 100 |
| Euler integration | 0.983 ms | 100 |

The RHS kernel accounts for about 54% of this timestamped compute sample. That is not 54% of application wall time. The 51 rebuilds reflect the loop's existing prepared-next-step scheduling. The sort executes 120 stage dispatches per rebuild at the padded 32,768-marker capacity.

The complete inventory and both diagnostic profiles are in [chrono-generated-profile.json](chrono-generated-profile.json). Static counts describe emitted sites, not how often a branch executes or the number of GPU instructions.

## Input and generated-code review

| CUDA entry / runtime operation | Finding and optimization constraint |
| --- | --- |
| `hashSelected` | Coordinate conversion contains corrected float divisions and floor. Replacing division by a reciprocal can change cell assignments at boundaries, so it needs a numerical equivalence proof, not a textual rewrite. |
| `findCellStartEndD` | Shared hash exchange and predicated early returns require barrier correctness. Its measured cost is small; deleting synchronization is not justified. |
| `neighborSearchNum` | Four nested loops traverse nearby cells/markers. Repeated position and parameter access is a candidate for alias-aware load reuse; counts and ordering must remain exact. |
| `neighborSearchID` | Repeats the search to fill the variable-length list. Combining count and fill requires a different allocation/execution plan, beyond a local WGSL trim. |
| `OriginalToSortedD` | Small scatter/index mapping kernel; little generated arithmetic to remove. |
| `reorderDataD` | Aggregate copying and property conversion dominate its source shape. Any store coalescing must preserve storage layout, inactive paths and diagnostics. |
| `UpdateActivityD` | Contains time/counter logic and promoted float comparisons. Only provably exact comparison cases were simplified. |
| `normalizeActivity` | Small normalization kernel. Scheduling and scan integration matter more than arithmetic simplification. |
| `fillActiveListD` | Prefix-sum scatter; preserve the active-list order and bounds. |
| `CopySortedToOriginalWCSPH_D` | Many source helpers disappear after reachability trimming. Remaining aggregate copies are candidates for scalar replacement, subject to aliasing and value-copy semantics. |
| `EulerStep_D` | Contains CFD/CRM and equation-of-state paths. Large generated source does not imply high active GPU cost. Exact promoted-float scaling was found in four sites. |
| `ApplyPeriodicBoundaryY_D` | Small boundary-update kernel. Wrap conditions and pressure changes must remain exact. |
| `Calc_Shifting_D<ShiftingMethod::XSPH>` | Neighbour traversal and vector accumulation; preserve accumulation order and corrected division. Reusing work across RHS/shifting is not automatically safe because intermediate state changes between passes. |
| `CfdAdamiBC_D` | Boundary-marker filtering, neighbour accumulation and inverse EOS. Some float-to-double comparisons can be narrowed exactly; actual double expressions remain. |
| `CfdCalcRHS_D` | Dominant compute kernel. The emitted application/helper functions contain 69 software-double call sites and 52 corrected-division call sites. Matrix branches, repeated aggregate loads and helper temporaries are the highest-value places for deeper dataflow analysis. Two promoted-float power-of-two multiplications were found. |
| Stable sort | 120 stages per rebuild are significant GPU and submission work. A future generic stage-fusion/local-memory optimization must preserve stable equal-key ordering; changing that order can change floating-point accumulation downstream. |
| Exclusive scan | Needed for active compaction and neighbour offsets. Preserve integer prefix sums, empty tails and total count; source trimming alone does not remove its synchronization requirements. |

The parser currently consumes `__restrict__` qualifiers without retaining a dedicated no-alias fact. Carrying that fact into a typed intermediate representation is a candidate for better load-hoisting and common-subexpression analysis. This does not establish that the backend currently fails to optimize those loads: read-only WGSL bindings and backend analysis already provide some information. Any improvement needs measurement.

## Implemented test-only optimizations

### Exact comparisons of promoted floats

The input contains tests such as `code < -0.5`, where `code` is float but the literal is double. Generated code promotes the float and uses the software-double comparator. When both operands are promoted floats, or one is a binary64 constant exactly representable as float, comparison can operate on the original float representations.

The replacement compares IEEE 32-bit representations, handling NaNs, signed zeros, infinities and subnormals explicitly. This avoids relying on native float comparison behavior for subnormals on hardware that may flush them. It does not narrow arbitrary double variables, actual double arithmetic, or inexact constants such as the tested `1e-12`.

There were 24 eligible sites across the Chrono shaders. The hardware edge test covered 30,384 comparisons, matching both the old GPU path and an independent binary64 CPU reference.

### Exact power-of-two scaling of a promoted float

`double(float_value) * 2.0` need not use a full significand multiplication. A finite nonzero float promoted to binary64 has a biased exponent between 874 and 1150. For an explicitly recognized positive power of two with exponent in [-512, 512], adjusting that exponent cannot overflow or become subnormal in binary64, and no rounding is needed.

The replacement preserves zeros/infinities and the original multiplier's NaN canonicalization. Arbitrary doubles, negative factors, nonpowers of two and shifts outside the proven range are refused. Six sites were found: four in Euler and two in RHS.

The hardware test compared all 64 output bits for 32,896 operations, including random float bit patterns and edge cases, and checked non-NaN results against an independent CPU reference.

### Explicit integer/boolean specialization after trimming

The previous specialization experiment preceded aggregate trimming. This experiment first simplifies aggregate field accesses, then supplies explicitly fixed integer/boolean uniform values through generated constant-returning helpers. The hypothesis is that the backend can more easily prune matrix/physics branches and their register requirements.

This is intentionally a different contract from the two exact dynamic rewrites. The test wrapper rejects changes to specialized scalars and rejects binding GPU counters to them. Float parameters remain dynamic. No application names or values are embedded in the specialization pass; the benchmark supplies its configuration explicitly.

## End-to-end results

Each run executes 3,000 full steps on the real RTX 5080, excluding compilation and output inspection. Final GPU completion is included. Each experiment compares final particle bits and every neighbour count against its baseline. Hardware runs are sequential.

| Experiment | Baseline mean | Optimized mean | Assessment |
| --- | ---: | ---: | --- |
| Exact comparisons, ABBA | 12.638 s | 12.630 s | No established gain |
| Power-of-two scaling, ABBA | 13.059 s | 12.640 s | Apparent gain required confirmation because baseline drifted |
| Power-of-two scaling, BAAB confirmation | 12.356 s | 12.460 s | Gain did not reproduce; no established benefit |
| Integer/boolean specialization after trim, ABBA | 12.443 s | 12.663 s | No gain; one optimized run was noticeably slower |

No runtime optimization should be promoted from these numbers. Removing software-level operations does not establish that the driver emits fewer instructions or that the changed operations dominate the critical path.

Reports: [comparison benchmark](wgsl-comparison-benchmark.json), [scaling benchmark](wgsl-scaling-benchmark.json), [reverse-order confirmation](wgsl-scaling-confirmation.json), [post-trim specialization](wgsl-posttrim-specialization.json), [comparison edges](wgsl-comparison-edges.json), [scaling edges](wgsl-scaling-edges.json), [combined numerical-pass regression](wgsl-numeric-suite.json).

## Validation and next priority

The combined comparison/scaling passes passed 248/248 real-GPU regression checks plus the dynamic-uniform/signed-zero check. All 18 experimental structural/numerical unit tests pass. Explicit specialization was tested separately in the four full-loop runs and in unit tests of its update/GPU-counter guards; the 248-case suite does not enable fixed configuration specialization. The suite wrapper intercepts runtime kernel creation, not raw shader creation in independent worker/device paths; the dedicated numerical edge tests directly exercise these new transformations.

The next substantial generated-code candidate is an alias-aware, typed dataflow pass for RHS: scalar replacement of local records/arrays, reuse of proven-invariant loads through helpers, and shorter temporary live ranges. Prioritize it with backend/register evidence. A source-level temporary count alone cannot prove register spilling. Stable-sort stage fusion is the other high-impact candidate, but it is a multi-dispatch optimization rather than a local shader text edit.

Reproduction commands:

```powershell
node --test tests/experiments/*.test.mjs
node scripts/test-wgsl-comparison-edges.mjs
node scripts/test-wgsl-scaling-edges.mjs
node scripts/profile-chrono-generated.mjs
node scripts/test-wgsl-trim.mjs 3000 .local/comparisons.json --comparisons
node scripts/test-wgsl-trim.mjs 3000 .local/scaling.json --scaling
node scripts/test-wgsl-trim.mjs 3000 .local/scaling-confirmation.json --scaling --reverse
node scripts/test-wgsl-trim.mjs 3000 .local/specialized.json --specialize
node scripts/test-wgsl-trim-suite.mjs .local/numeric-suite.json --numeric
```
