# NVIDIA sample compatibility explorer

Open [the audit explorer](./) for diagnostic details. Each verified kernel has a
CUDA/WGSL sandbox link and an explicit numerical fixture runner.

Scope: every sample directory with a README under `cpp/` and `python/` at
NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.
The audit scans direct source files for standalone global void kernels. It is
not a complete C++ preprocessor or a claim to compile every template instance.

The 21 translated entries were run separately with NVCC on RTX 5080 and with
WebGPU on a real NVIDIA adapter. Fixtures use 256 elements or a 32×32 grid,
with 64×64 matrices for the two transpose kernels and 17 vectors of 1,537
elements for the scalar-product kernel;
every output component is compared with an independent CPU reference (absolute
tolerance 0.000003). These are small correctness checks, not performance results
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
node scripts/prepare-nvidia-checks.mjs
nvcc -O3 -std=c++17 -arch=native -Xcompiler /Zc:preprocessor .local/nvidia-checks/check.cu -o .local/nvidia-checks/check.exe
.local/nvidia-checks/check.exe
node scripts/test-nvidia.mjs
node scripts/test-nvidia-transpose.mjs
node scripts/test-nvidia-scalar.mjs
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
