# NVIDIA sample compatibility explorer

Open [the audit explorer](./) for diagnostic details. Each verified kernel has a
CUDA/WGSL sandbox link and an explicit numerical fixture runner.

Scope: every sample directory with a README under `cpp/` and `python/` at
NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.
The audit scans direct source files for standalone global void kernels. It is
not a complete C++ preprocessor or a claim to compile every template instance.

The 26 translated entries were run separately with NVCC on RTX 5080 and with
WebGPU on a real NVIDIA adapter. Fixtures use 256 elements or a 32×32 grid,
with 64×64 matrices for the two transpose kernels and 17 vectors of 1,537
elements for the scalar-product kernel;
every output component is compared with an independent CPU reference (absolute
tolerance 0.000003, except Black–Scholes below). These are small correctness checks, not performance results
or exhaustive numerical validation. The browser records its actual adapter.

These are **isolated kernel stages**. Passing the ocean heightmap stage does
not implement cuFFT. Passing a peer-copy kernel does not test peer access.
Passing the Tegra sine-wave kernel does not run Jetson/EGL host code on Windows.
Full native program attempts are recorded separately, including their exit
status, timeout and log. Exit code zero alone does not prove correct output.

Reproduce from a checkout of that NVIDIA revision in `.local/nvidia-audit`:

```
node scripts/audit-nvidia.mjs
node scripts/prepare-nvidia-transpose.mjs
node scripts/prepare-nvidia-scalar.mjs
node scripts/prepare-nvidia-blackscholes.mjs
node scripts/prepare-nvidia-matrixmul.mjs
node scripts/prepare-nvidia-scan-update.mjs
node scripts/prepare-nvidia-atomic-cas.mjs
node scripts/prepare-nvidia-checks.mjs
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor .local/nvidia-checks/check.cu -o .local/nvidia-checks/check.exe
.local/nvidia-checks/check.exe
node scripts/test-nvidia.mjs
node scripts/test-nvidia-transpose.mjs
node scripts/test-nvidia-scalar.mjs
node scripts/test-nvidia-blackscholes.mjs
node scripts/test-nvidia-matrixmul.mjs
node scripts/test-nvidia-scan-update.mjs
node scripts/test-nvidia-atomic-cas.mjs
```

Run the server first (`npm start`) for browser checks. The browser script uses
installed Edge, or `CW_CHROMIUM`; it requires an NVIDIA hardware adapter and
never requests a software adapter. Full upstream build uses CMake with
`CMAKE_CUDA_ARCHITECTURES=120`; `scripts/run-nvidia-native.mjs` records executable
attempts with a 12-second time limit (120 seconds for UnifiedMemoryPerf and
tileMatmulAutotuner). Run `node scripts/report-nvidia.mjs` after the checks to
merge execution results into the audit. Re-running the inventory resets its
statuses; keep execution evidence in the separate reports.

CUDA source and translated WGSL retain NVIDIA's BSD-3-Clause license. Original
fixture, explorer and audit code are MIT. See `../../THIRD_PARTY_NOTICES.md`.

The transpose follow-up imports the exact coalesced and padded kernel bodies,
their original numeric tile constants and namespace alias. It does not add full
C++ support for every other function in the upstream desktop file. The initial
direct-source probe errors remain in the audit as baseline evidence; the
follow-up GPU checks record the newly supported extracted kernels.

`transpose-native.cu` checks both variants at 32×32, 64×96, 96×64 and 128×128,
including 16 guard values. Build it with the same NVCC flags above.
`reports/nvidia-transpose-native.txt` and `reports/nvidia-transpose.json` record
the native and WebGPU results; the latter also checks divergent-barrier rejection.

`scalar-native.cu` checks the unchanged scalar-product kernel with 1, 33, 1,537
and 4,096 elements per vector, including more vectors than blocks and 16 output
guard values. It also checks signed/unsigned 24-bit multiplication with 121
boundary pairs. `reports/nvidia-scalar-native.txt` and `reports/nvidia-scalar.json`
record native CUDA and real WebGPU checks. The browser also verifies overflowing
literal operands. All arithmetic references are separate from the sandbox runner.

The compiler accepts bounded direct function-forwarding macros, including the
original `IMUL` declaration. Every parameter must be forwarded once and in order;
chains are limited to 32 calls. Variadic macros, expression substitution, token
pasting, stringification and recursion remain unsupported. The source importer
preserves supported forwarding directives and their original line positions.

Black–Scholes imports the complete upstream `BlackScholes_kernel.cuh` unchanged,
including both helpers and its 128-thread launch bound. `blackscholes-native.cu`
and `scripts/test-nvidia-blackscholes.mjs` check 0, 2, 10, 258, 259 and 1,024
options, both output arrays and 16 trailing guard values. The original kernel
processes pairs; the unpaired odd option remains untouched. Native references
use double-precision `erfc`; browser references independently integrate the
normal density using Simpson's rule. The error limit is 0.0002 absolute plus
0.00002 relative, accommodating the sample's normal-CDF approximation and WGSL
math differences. These checks do not establish general fast-math equivalence.
Results are in `reports/nvidia-blackscholes-native.txt` and
`reports/nvidia-blackscholes.json`. Build the native harness with the same NVCC
flags above. Preset buffers use generic fill patterns with optional `scale` and
`offset` to provide positive prices and times; all option calculations run in
the compiled shader. Select either result buffer in sandbox settings.

The matrixMul follow-up retains the original `template <int BLOCK_SIZE>`
declaration and complete kernel body. Entries `MatrixMulCUDA<16>` and
`MatrixMulCUDA<32>` select the specialization explicitly; the compiler does not
infer template values from the launch dimensions. Both use square thread blocks
matching the tile, with positive matrix dimensions divisible by that tile.
The unchanged kernel does not handle arbitrary edge tiles. `matrixmul-native.cu`
and `scripts/test-nvidia-matrixmul.mjs` cover eight cases: square and rectangular
matrices, one and multiple inner tiles, multiple output blocks, and 16 trailing
guard values. Binary-fraction inputs permit exact comparison with independent
CPU matrix multiplication. Native and browser results are recorded in
`reports/nvidia-matrixmul-native.txt` and `reports/nvidia-matrixmul.json`.
Compile the native harness using the NVCC flags above. These are correctness
checks, not evidence that either tile size is faster.

The scan follow-up imports only the unchanged `uniformUpdate` kernel plus its
namespace alias. Its `uint4` data receives a shared block offset; the two earlier
scan stages still require shared-memory pointers, helper barriers and other
unsupported helper constructs. No complete scan is claimed. Native harnesses
supply `using uint = unsigned int`, the alias normally supplied by NVIDIA helper
headers. `scan-update-native.cu` and `scripts/test-nvidia-scan-update.mjs` check
four block/thread configurations, values above 2^24, unsigned wraparound and 16
guard values. The browser additionally checks signed/unsigned vector constructors,
helper returns and shared scalar atomics. Results are recorded in
`reports/nvidia-scan-update-native.txt` and `reports/nvidia-scan-update.json`.
Build the native harness with the NVCC flags above. The sandbox uses generic
integer buffers and the compiled shader; its preview performs no CPU scan work.

The atomic follow-up preserves the complete `cas_atomic` kernel body and original
`NUM_THREADS` / `ARRAY_SIZE` constants from `simpleAtomicIntrinsics`. Only this
intrinsic-based kernel is imported; the CCCL atomic_ref variants remain outside
the supported subset. `atomic-cas-native.cu` and
`scripts/test-nvidia-atomic-cas.mjs` check 32, 384, 4,352 and 1,000,000 active
threads updating ten counters, including extra launched lanes beyond the source
limit and 16 guard values. Browser checks additionally verify signed/unsigned
CAS success/failure return values, shared scalar CAS and do-while continuation.
The emitter handles possible spurious failure of WGSL weak compare-exchange by
retrying while the observed value still equals the comparison value and no
exchange occurred. Reports are `reports/nvidia-atomic-cas-native.txt` and
`reports/nvidia-atomic-cas.json`; native build flags are the same as above.
