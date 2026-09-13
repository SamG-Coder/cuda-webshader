# Chrono SPH dam-break port: GPU preparation, boundaries and forces

This is an in-progress compiler port, not a runnable water showcase.
The GPU chain now connects activity selection, normalized compaction, original
marker IDs, grid sorting, original property reordering, cell ranges and neighbour
lists, Adami wall-pressure evaluation and CFD force derivatives. Particle shifting
and time integration are still pending.

Upstream: https://github.com/projectchrono/chrono at
`a92c6f72f422fbcafe0b37125d4070cb6a3b5803`.
The original `src/demos/fsi/sph/demo_FSI-SPH_DamBreak.cpp` and solver are unchanged.
Chrono's BSD terms are reproduced in `THIRD_PARTY_NOTICES.md`.

## Native reference

Built the original `demo_FSI-SPH_DamBreak` target with CUDA 13.3, native GPU
architecture, MSVC 19.51, Release, single precision, and Eigen revision
`549bf8c75b6aae071cde2f28aa48f16ee3ae60b0`. FSI and FSI-SPH are enabled;
FSI-TDPF, tests and optional visualization are disabled. The out-of-source build
is `.local/chrono-build`; the original source is `.local/chrono-upstream`.
MSVC requires `/Zc:preprocessor` for this toolchain.

Executed:

```text
demo_FSI-SPH_DamBreak.exe --no_vis --output --output_fps 2 --quiet
```

The process exited successfully after the original ten simulated seconds
(approximately 100,000 RK2 steps of 0.0001 seconds), taking 157.931 wall seconds.
Only logging, visualization and snapshot frequency were set on the command line.
The simulation contains 16,731 fluid particles. All 20 saved particle snapshots
have the same particle count and finite values in all ten columns. Snapshots
cover t=0 through t=9.5; the last saved snapshot is not the final t=10 state.
`chrono-dambreak-native.json` records per-frame ranges and SHA-256 hashes.
The CSV files are local reference data under
`.local/chrono-native-output/CFD_WCSPH_RK2_ps1/particles/`.
Finite output and process success do not establish agreement with an experimental
fluid benchmark; they establish the native implementation's reference run.

## First compiler issue fixed

`findCellStartEndD` lets individual lanes return before `__syncthreads()`.
WGSL generation succeeded initially, but real NVIDIA WebGPU validation rejected
the non-uniform barrier. `predicatedReturns: true` now lowers supported
straight-line kernel phases: returned lanes still reach top-level barriers,
while subsequent conditions, initializers and memory operations execute only
for active lanes. The CUDA fixture remains unchanged. No uniformity diagnostics
are disabled. Nested barrier control flow, nested returns, non-scalar local
lifetimes and top-level loops are rejected by this bounded option.

`OriginalToSortedD` and `findCellStartEndD` execute on real NVIDIA Blackwell
WebGPU. They agree exactly with the original native CUDA kernels and an
independent cell-span/permutation oracle for 2,936 values across particle counts
0, 1, 31, 127, 128, 129, 255, 256, 257 and 1027. Output sentinels verify that
inactive lanes do not write. This covers full and partial blocks and cell spans
crossing block boundaries; it is not a complete neighbour-search or SPH step.

To regenerate the native kernel reference from a Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-neighbors-native.cu -o .local/chrono-neighbors-native.exe
.local/chrono-neighbors-native.exe > reports/chrono-neighbors-native.json
```

Run `npm test`, `npm run compile`, `npm run build`, then `node scripts/test-gpu.mjs`
with `CW_SOFTWARE_GPU=0` and `CW_CHROMIUM` pointing to the installed Edge executable.
The GPU suite includes the native-reference comparison.

## Remaining for the water showcase

Handle namespace-aware header ingestion and the solver's active/extended marker
selection, gather all required particle properties, then add ADAMI boundary
forces, WCSPH pressure/viscosity and RK2 integration.
Compare intermediate buffers and evolved particle states against native CUDA.
Only then wire the full solver to a sandbox preview and add its showcase card.
The native CSV sequence must not stand in for browser computation.

## Scoped enum and constant-record support

All 16 scoped enums (52 values) are extracted unchanged into
`tests/chrono-enums.cu`. The compiler supports 32-bit signed or unsigned scoped
enums, qualified members and bounded integer constant expressions. Unsigned
expression arithmetic retains its wraparound and logical-shift semantics.
Boolean fields in constant records are represented by validated u32 uniforms
and converted to WGSL booleans. Plain record declarations now allow up to 256
fields; constant aggregates retain their separate component limit.

The original enum values and all four combinations of two boolean parameters
match native CUDA exactly. A separate 121-field constant-record GPU probe
checks first/last field access and boolean branch selection. There are 218
comparisons in this stage (216 native-backed enum/flag values and two large-record
checks). The full local suite passes 688 unit tests and 226 NVIDIA GPU tests.

The next stage below adds the full `ChFsiParamsSPH`, including its
`double pressure_height` field. General namespace/header ingestion remains
unsupported; these fixtures extract the original declarations and helpers into
one device translation unit without changing their bodies.

Native enum reference regeneration, from the Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-enums-native.cu -o .local/chrono-enums-native.exe
.local/chrono-enums-native.exe > reports/chrono-enums-native.json
```

## Original grid-position and grid-hashing stage

The full original parameter record now compiles, including its binary64 field.
Double fields in plain records are represented by two exact u32 uniform words
(`.lo` and `.hi`) and use the existing integer-based binary64 expression code.
They are not demoted to float32. Standalone double locals, double buffers and
record storage ABIs containing doubles remain explicitly unsupported.
The original `__constant__ static` spelling and CUDA `floor(float)` overload
are also accepted. The latter was checked with a native compile-time type assertion.

`tests/chrono-params-native.cpp` executes the unchanged dam-break setup through
`sysFSI.Initialize()`, then captures `GetParams()` and the actual initial
particle positions. It reuses the upstream demo compiler flags, including
Eigen/AVX2 alignment settings. The native parameter ABI is 608 bytes; the
browser receives 157 named 32-bit parameter words instead of a raw struct copy.

The original `calcGridPos`, `reduceGridIndex`, and `calcGridHash` helpers match
native CUDA for all 16,731 initial particles plus eight boundary/far-outside
positions. Nine runs cover the actual dam-break periodic flags followed by all
eight combinations of periodic/clamped axes. Every grid-coordinate component
and final hash agrees exactly: 602,604 integer comparisons. Real3 input stride
is checked as 12 bytes. A precision test sets the double field to 1+2^-40 and
verifies that subtracting 1 retains 2^-40.

This is position hashing, not complete neighbour search or fluid integration.
No water showcase has been added. Current local validation: 690 unit tests and
227 real NVIDIA WebGPU tests passed.

Regenerate from the configured pinned native build:

```text
node scripts/build-chrono-capture.mjs
.local/chrono-build/bin/chrono-params-capture.exe --quiet --no_vis
node scripts/prepare-chrono-hash-reference.mjs
```

Then, from a Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-hash-native.cu -o .local/chrono-hash-native.exe
.local/chrono-hash-native.exe
```

The native harness writes `reports/chrono-hash-native.bin`; the GPU suite
checks it using the captured `reports/chrono-params.json` and particle inputs.

## Original neighbour-list construction

The native capture now copies the initialized marker position/radius buffer,
including all 30,327 markers: 16,731 fluid and 13,596 BCE boundary markers.
`GetPositions()` establishes the buffer length; `GetMarkerDeviceView()` exposes
the same underlying device storage. The host capture reuses the original setup
and compiler flags and links the configured CUDA runtime for this copy.

`tests/chrono-search.cu` retains `neighborSearchNum`, `neighborSearchID`,
`Distance`, `Modify_Local_PosB`, `MinimumImageShift`,
`MinimumImageShiftMultiPeriod` and the required original constructors/operators.
Only the hashing/index-initialization and position-permutation wrappers are MIT
harness code. The density argument is unused by both original search kernels
and is backed by an unused one-record allocation in both harnesses.

The browser performs hashing, stable key/index sorting, position gathering,
cell-range construction, neighbour counting, exclusive scanning and neighbour-ID
generation on the GPU. It reads back only the four-byte count used to allocate
the neighbour buffer, matching the native host allocation dependency. Reference
arrays are read afterward for validation, not used to calculate neighbour IDs.

All hashes, sorted indices, cell ranges, counts, offsets and neighbour IDs match
the native CUDA/Thrust pipeline exactly: 1,027,953 values, including 794,643
neighbour entries. This test runs the complete search pipeline on all initialized
markers. It does not yet reproduce the full solver's active/extended-marker
selection or advance pressure, velocity or position. No water showcase is added.

Compiler additions: bounded identifier macro aliases, including late binding and
forwarded calls; acceptance of the original `__declspec(noinline)` annotation
(WGSL controls inlining); the float-only CUDA `rint` overload; and unambiguous
scalar conversions for overloaded helpers. Ambiguous overloads, alias cycles
and unsupported annotation kinds are rejected. Identifier aliases inside the
separate object-expression macro facility explicitly require preprocessing.

Current local validation: 693 unit tests and 228 real NVIDIA GPU tests passed.
Regenerate after building the pinned Chrono dependency:

```text
node scripts/build-chrono-capture.mjs
.local/chrono-build/bin/chrono-params-capture.exe --quiet --no_vis
node scripts/prepare-chrono-search-reference.mjs
```

Then, in a Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-search-native.cu -o .local/chrono-search-native.exe
.local/chrono-search-native.exe
```


## Activity-selection prerequisites: exact counters and launch time

The original `Counters` declaration (22 native `size_t` fields, 176 bytes) now
compiles as a constant record without narrowing its values. Each field is bound
as low/high unsigned words. An explicit by-value `double time` kernel parameter
uses the same exact two-word representation and the existing binary64 expression
implementation. No native fp64 WebGPU feature is assumed.

`tests/chrono-counter-time.cu` retains the upstream declaration unchanged and
adds an MIT validation probe. The native harness supplies 12 controlled cases,
including values above 2^32 and 2^53, the unsigned 64-bit maximum, times 2^-40
above/below 0.5, signed zero, infinities, and NaN. These are type/ABI probes, not
captured activity-selection outputs or simulated water frames. The independent
CPU oracle also reconstructs exact integer and binary64 values from word pairs.

The next blocker remains native-layout `ActiveDomain` buffers (a one-byte bool
followed by four Real3 values). Full original `UpdateActivityD` execution,
active-list compaction, pressure/force evaluation and time integration are still
pending. No new water showcase is added at this stage.

Reproduce the native counter/time reference in a Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-counter-time-native.cu -o .local/chrono-counter-time-native.exe
.local/chrono-counter-time-native.exe > reports/chrono-counter-time-native.json
```

Validation for this stage: 696 unit tests, 229 real NVIDIA GPU tests, and all
1,152 counter/time probe values match native CUDA exactly. Compile checks pass.


## Original activity-selection kernel

`tests/chrono-activity.cu` now compiles and executes the original `UpdateActivityD`,
`checkActivityD`, and `inAABB` functions without editing their bodies. The native
harness verifies `ActiveDomain` is 52 bytes with fields at offsets 0, 4, 16, 28,
and 40. Read-only buffers are bound as raw words and decoded to computational
records, preserving one-byte booleans and ignoring padding. Records are decoded
at each source read, including helper pointer offsets. Writable packed records,
unsupported field types and record strides not divisible by four remain explicit
errors. Native float3 fields use their CUDA alignment rather than WGSL storage
alignment. GPU-resident 32-bit scalar buffer elements can also bind helper
references, allowing the original helper to update activity outputs directly.

The native reference runs 14 cases. The first uses the 30,327 captured initialized
marker positions and captured dam-break parameters, with zero initial velocities
and explicitly seeded fluid/BCE type codes from the known fluid-prefix count.
It is an isolated kernel check, not a capture of Chrono's complete internal state.
Thirteen controlled cases use 198 markers each, covering fluid/helper/ghost and
BCE type codes, inverted boxes, all three domain arrays, inclusive boundaries,
extended-only activity, nonperiodic outside-domain removal, eight periodic flag
combinations, settling-time thresholds separated by 2^-40, and empty active-domain
lists. Domain padding bytes are deliberately 0xA5, including padding after false
booleans. Native CUDA and WebGPU agree on all 164,505 activity/extended-activity
and velocity words. Nine further values check adjacent bool bytes, float3 fields,
const local references, pointer offsets and aliased scalar output references
against native CUDA.

This advances marker selection only. Connecting selection and active-list
compaction to the neighbour pipeline, pressure/force evaluation and integration
remains necessary before the original solver can advance water in the sandbox.
No water showcase is added yet.

Regenerate after capturing the pinned native parameters and search input:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-activity-native.cu -o .local/chrono-activity-native.exe
.local/chrono-activity-native.exe
```

Validation for activity selection: 699 unit tests and 230 real NVIDIA GPU tests
passed, including the existing Sobel, neighbour, quadtree and rendering checks.


## Activity scan and compaction: native defect and explicit adaptation

The original `ActivityScanOp` is retained unchanged in
`tests/chrono-activity-scan.cuh`. It computes `op(a,b) = a + max(b,0)` and is not
associative over activity flags containing -1. For example,
`op(op(1,-1),1) = 2`, while `op(1,op(-1,1)) = 1`. Parallel scans require an
associative operator; see NVIDIA's [DeviceScan contract](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html).

On this CUDA 13.3 / NVIDIA Blackwell toolchain, running the unchanged native scan
on the activity fixtures produces 1,464 wrong prefix values in eight of the
14 cases. Each affected case has one duplicate write offset among selected
markers and a final count smaller by one. The original initialized dam-break
case has only positive flags and passes. The baseline prefix arrays, counts,
and collision diagnostics are retained in `chrono-compact-native.bin` and
`chrono-compact-native.json`. This is an upstream scan-domain problem, not an
observed defect in the translated activity kernel. The defective native compact
writes are not executed; collisions are established by inspecting native prefix
positions for markers the unchanged `fillActiveListD` would write.

The host adaptation is explicit: a GPU kernel maps inactive activity flags to
zero before the scan. The native reference then invokes the unchanged original
scan functor on those normalized values, where it is associative. WebGPU uses
the existing unsigned exclusive scan on the same normalized values. This does
not claim to reproduce the faulty negative-flag baseline or silently alter the
original functor. The original `fillActiveListD` still receives the original
activity flags, and only writes markers whose flag is 1.

The validation pipeline now executes original `UpdateActivityD` -> normalize
extended flags -> GPU exclusive scan -> original `fillActiveListD` -> gather
selected positions. Activity flags stay on the GPU between stages. Only the
four-byte selected count is read to size and launch the gathered output, matching
the native host's need for a selected count. There are no intermediate array
uploads or CPU physics. The zero-selected case is included. Native-normalized
prefixes, full compact index buffers (including untouched tail slots), and
selected position words match exactly across all 14 cases. The original
activity outputs are also rechecked after the pipeline.

Remaining: connect selected positions and their original marker IDs to neighbour
construction, then implement the complete property/force/integration sequence.
The water solver is still not ready for a showcase.

Reproduce with the existing activity references and a Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-compact-native.cu -o .local/chrono-compact-native.exe
.local/chrono-compact-native.exe
```

Validation: 700 unit tests, 231 real NVIDIA GPU tests, and 193,110 exact
native-normalized compaction values passed. Fourteen cases read a total of
56 intermediate bytes, with no intermediate array uploads.


## Selected markers connected to neighbour construction

The pipeline now continues directly from the GPU-produced active list into grid
hashing, sorting, position gathering, original cell-range construction, original
neighbour counts, neighbour offset scan and original neighbour ID construction.
The hash adapter retains original marker IDs, so gathering after the sort reads
from the original position buffer even when compaction removed markers.

`tests/chrono-selected-native.cuh`, invoked by the native compaction harness,
consumes the device compact list and positions without copying those arrays to
the CPU between stages. `tests/chrono-selected-gpu.js` does the same after the
original activity kernel and explicit positive-flag normalization. Both use the
existing original grid helpers and neighbour kernels. Hash/gather entry points
remain explicit MIT data-adapter kernels: the complete upstream `calcHashD`
entry point's diagnostic `printf` paths and volatile boolean error flag are not
claimed as supported by this milestone.

All 14 cases match in 1,199,815 hash, original-index, cell-range, count, offset,
neighbour-ID and sorted-position words. They contain 807,171 neighbour entries
in total. The initialized dam-break case retains all 30,327 markers and its
794,643 neighbours; controlled cases retain 180, 114 or zero markers. Controlled
cases use a consistent 10-by-10-by-10 grid with 0.2 cell widths in their 2-unit
box, while the initialized case retains captured Chrono parameters.

Only the selected-marker and neighbour counts return to the CPU for allocation:
eight bytes per case, 112 bytes across all cases. Validation readbacks happen
after the associated stages and are excluded from that transfer measurement.
There are no intermediate array uploads or CPU physics. The activity and
compaction references are also rechecked after the neighbour chain runs.

Reproduce the combined native references with:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-compact-native.cu -o .local/chrono-compact-native.exe
.local/chrono-compact-native.exe
```

Remaining before a water showcase: complete the original error/diagnostic paths,
reorder all needed marker properties (not just positions), and run the pressure,
force and time-integration stages against the native solver. This chain is
preprocessing for the solver and does not advance fluid motion.

Validation for the connected chain: 700 unit tests, 232 real NVIDIA GPU tests,
and compile checks passed.


## Original property reorder and bounded diagnostic capture

`tests/chrono-reorder.cu` retains the original `reorderDataD` and `IsFinite(Real3)`
functions unchanged, including their diagnostic calls. WebGPU compilation selects
the original device branch with `defines: { __CUDA_ARCH__: 1 }`; native nvcc
selects its actual CUDA architecture. The compiler now supports scalar float
`isfinite` through exponent-bit classification and captures standalone integer
`printf` diagnostics into an explicit GPU buffer. It does not drop error paths.

The native fixture supplies 521 controlled input records and a noncontiguous
selection of 259 original IDs. Four cases cover CFD and CRM, each with finite
inputs and then deliberately non-finite inputs. All 21,756 output words match
native CUDA exactly, including positions, velocities, density/pressure/viscosity,
activity, and stress-related fields. CFD leaves the three stress-related outputs
at their initialized sentinel values; CRM reorders them. Native and WebGPU emit
the same three diagnostic messages in total. The original text says 'NAN' even
for the deliberately injected infinity, and that text is preserved.

The kernel uses 15 property/index buffers plus the diagnostic buffer, reaching
16 storage bindings. Additional GPU validation covers float finite classification,
signed/unsigned integer formatting, percent escaping, and capture overflow: five
attempted messages with capacity two yield two records and three dropped events.

Current diagnostic scope is explicit: literal formats with %u, %d and %%, up to
eight integer arguments, up to 64 format strings, standalone statements only.
`diagnosticCapacity` defaults to 64 records and accepts 1..65536. Metadata exposes
the required `cw_printf_storage` buffer and format table; callers allocate and
clear that buffer and call `decodeDiagnostics` after GPU completion. Capture
order across lanes is unspecified, and attempted/dropped counters are 32-bit.
Float formatting, arbitrary strings, and printf return values remain unsupported.
In particular, full upstream `calcHashD` diagnostic/error-flag support is pending.

This is a standalone original-kernel/property-layout check. Connecting the reordered
properties to the existing activity/neighbour chain and then force/integration
stages remains necessary; no water showcase is added yet.

Reproduce in a Visual Studio developer shell:

```text
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor tests/chrono-reorder-native.cu -o .local/chrono-reorder-native.exe
.local/chrono-reorder-native.exe > .local/chrono-reorder-native.log
node scripts/prepare-chrono-reorder-messages.mjs
```

Validation: 703 unit tests and 233 real NVIDIA GPU tests passed, including
21,756 native-exact property words and the native diagnostic messages.

## Connected native marker properties

The dam-break case now starts from `GetProperties()` and `GetVelocities()` captured
immediately after the unchanged native demo initialization. The earlier activity
fixture used placeholder density/pressure/viscosity (1000/0/0.001), which sufficed
for marker-type selection but was unsuitable as fluid-solver input. The actual
capture has density 1000–1003.822021484375, pressure 0–38220, and viscosity 5.
`prepare-chrono-property-reference.mjs` checks capture size, finite values,
parameter equality and exact position equality before preparing the fixtures.
The public properties API supplies three components; the host fixture restores
only the fourth marker-type component from this demo's initialized ordering:
16,731 fluid markers (-1), then 13,596 fixed boundary markers (0).

The connected GPU chain now runs the unchanged `OriginalToSortedD` and
`reorderDataD`, consuming live activity-stage velocity, property and activity
buffers. Neighbour kernels receive the resulting sorted property buffer. Native
CUDA consumes its corresponding activity reference outputs. Both preserve the
original global marker IDs and leave inactive inverse-map entries at UINT_MAX.
The native-only `chrono-selected-reorder.cuh` retains the original functions;
the browser uses the already tested original reorder and neighbour source files.

Across 14 cases, all 1,487,332 output words agree exactly with native CUDA,
including sorted positions, velocities, properties, activity and the inverse map.
The 807,171 neighbour entries are unchanged. Intermediate allocation readback
remains eight bytes per case (112 total), with no intermediate array uploads or
CPU physics. Original reorder diagnostics are empty for these finite inputs.
This connected fixture asserts CFD; separate tests cover CRM stress reordering.

The next native solver operation for this captured configuration is Adami boundary
conditions, then CFD derivatives and particle shifting (enabled in this demo).
Pressure/force evaluation and RK2 time integration are not yet connected, and
this remains an in-progress port rather than a runnable water showcase.

## Boundary-condition compiler work

The next original kernel, `CfdAdamiBC_D`, calls smoothing-kernel and inverse
state-equation helpers that use C++ switch statements. The compiler now parses
and emits these statements, preserving C++ fall-through through repeated case
suffixes in WGSL. A separate unchanged CUDA control-flow fixture compares signed
and unsigned selectors, default in the middle, stacked labels, conditional break,
case scopes, returns, nested loops/switches and outer-loop continue.

The original boundary fixture now parses past those helpers and stops at its
`volatile bool* error_flag` parameter. This identifies the next ABI/parser gap;
the boundary calculation has not yet executed through WebGPU.

## Packed boolean error flags

The compiler now accepts original `volatile bool*` kernel parameters and preserves
CUDA's one-byte bool storage stride. Boolean stores normalize to 0/1 and atomically
update only their byte, preserving adjacent marker flags and allocation padding.
Volatile bool reads use atomic loads. The native/GPU fixture covers 1,025 lanes,
eight repeated dispatches, indexed reads/writes, a dereferenced single error flag,
and untouched guard bytes. The CPU oracle requires Uint8Array for these buffers.

The unchanged Adami boundary signature now compiles past the error flag. Including
its original Real3 math dependencies exposes the next frontend gap: the free
compound-assignment overload `operator+=(Real3&, Real3)`. Boundary physics and
simulation time advancement remain unverified and unavailable as a showcase.

## Original Adami boundary calculation connected

`CfdAdamiBC_D`, its smoothing and inverse-EOS helpers, and its Real3 math bodies
now compile unchanged. Free void compound-assignment operators accept a mutable
class reference and a value/const-reference operand. User-defined `length` for a
record follows the original helper instead of the vector-only WGSL builtin.

The real NVIDIA GPU test calls the complete preparation chain, copies its sorted
properties and velocities on the GPU to retain the preparation reference, then
runs Adami with live neighbour offsets/IDs and positions. Fixed-wall acceleration
is zero for this dam-break fixture. There are no intermediate array uploads or
readbacks during this boundary calculation. Native CUDA uses the matching
preparation snapshot in `chrono-adami-input.bin`; the original native kernel
updates 902 boundary records and leaves all 16,731 fluid records unchanged.

All 212,289 floating values match with relative tolerance 2e-5 and absolute
1e-5, except pressure uses absolute 2e-4. Fluid properties, viscosity and marker
type are checked exactly. Maximum absolute difference is 0.00390625 at larger
pressure magnitude; these results are numerical agreement, not bitwise equality.
The initially failing near-zero pressure belongs to sorted marker 23513: native
approximately -0.09500281 versus WebGPU -0.09510522. The independent double
reference is -0.09508536, from opposing contributions +980.000061 and -980.095146.
`scripts/diagnose-chrono-adami-cancellation.mjs` reproduces this cancellation
analysis. CUDA `--fmad=false` changed some values but did not remove this one
near-zero discrepancy; the accepted reference remains the default native build.

This is a verified boundary stage, not a complete time step. CFD force derivatives,
configured particle shifting and RK2 integration remain before a water showcase.

## Original CFD force kernel connected

`CfdCalcRHS_D` and its original helper bodies now compile and execute. The complete
source includes gradient/laplacian corrections and the symbolic 6x6 inverse; those
correction options are disabled in the captured native demo configuration. They
were compiled, but their numerical results are not covered by this default-case
comparison. Active settings include cubic-spline smoothing, laminar viscosity and
delta-SPH. No physics branch was replaced with a stub or a different equation.

The compiler changes cover unary record operators, CUDA float math overloads,
`__inline__`, mixed storage/local-array pointer helper calls, and iterative emission
of long arithmetic chains. Local array specialization preserves array length and
reference identity through helper forwarding. A separate native/GPU fixture checks
128 exact output words, local arrays of lengths 8 and 12, scalar pointer forwarding,
and a 900-term cancellation expression whose original evaluation order matters.

The connected comparison uses GPU-produced neighbours and boundary properties,
then checks all 242,617 output words: acceleration/density derivatives, free-surface
IDs, position divergence, Courant and acceleration timestep estimates, and the error
flag. No intermediate CPU transfer is used by the force calculation. Original
invalid-derivative diagnostics remain active and no messages were reported.

The numerical limits are specific to this initialized-state comparison:
- Acceleration components: absolute 4e-4 plus relative 2e-4.
- Density derivative: absolute 1e-2 plus relative 2e-4.
- Position divergence: absolute 1e-5 plus relative 2e-5.
- Acceleration timestep: absolute 1e-5 plus relative 3e-3.
- Free-surface IDs, Courant timestep and error flags: exact.

Observed connected maxima are approximately 2.75e-4 in acceleration, 8.56e-3
in density derivative, 1.67e-6 in divergence and 3.04e-3 in acceleration timestep.
The isolated comparison supplies bit-identical native boundary properties; its
maximum density-derivative error falls below 9.51e-6. This separates amplification
of boundary-density roundoff by delta-SPH from arithmetic inside the force kernel.
Independent float64 diagnostics for representative worst components show net
accelerations near zero formed from contributions with absolute sums around
577–754. Both native CUDA and WebGPU differ slightly from that higher-precision
reference. `diagnose-chrono-rhs-roundoff.py` reproduces these diagnostics.

These are numerical tolerances, not a bitwise-force equivalence claim or a
long-duration accuracy claim. A native-matched full integration step and later
trajectory comparisons remain necessary. Particle shifting and RK2 integration
are still required before adding a running Chrono water showcase.
