# Chrono SPH dam-break port: native reference, neighbours and activity selection

This is an in-progress compiler port, not a runnable water showcase.

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
