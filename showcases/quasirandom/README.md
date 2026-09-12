# NVIDIA quasirandom cube

[Open in sandbox](../../sandbox.html?example=quasirandom).

This showcase retains the original `quasirandomGeneratorKernel`, table declaration and multiplication macro from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/quasirandomGenerator/quasirandomGenerator_kernel.cu`. The original three-dimensional workload uses N=1,048,576, 128 blocks, and 128 × 3 threads per block. It generates all 3,145,728 coordinates.

The original CPU table initializer supplies the 3 × 31 Niederreiter direction table, captured in `reports/quasirandom-table.bin` and copied verbatim into the editable pipeline scalar settings. The browser runs the unchanged CUDA kernel; native output is only a test reference. The separately labelled MIT `packQuasirandomPoints` helper copies coordinate planes into a float4 rendering buffer without scaling or resampling. All points render from the shared GPU buffer as small sprites. CPU inspection sets camera bounds; it does not generate or upload replacement point positions.

The sequence is deterministic. Edit `seed` in the first pipeline step and compile again to generate a different segment. The static point cloud does not animate simulation steps; drag to orbit, scroll to zoom.

## Verification

The unchanged complete native program passes its CPU comparisons: QRNG L1 norm 7.275964e-12, inverse CND L1 norm 1.101623e-7. The new showcase covers the QRNG stage. The inverse CND stage is not part of this point-cloud pipeline.

Real NVIDIA Blackwell WebGPU matches all 3,145,728 default coordinate bits. Additional 1,025-point cases at seeds 1 and 4,294,967,280 cover nonuniform tails and seed wrap. The sandbox test verifies every packed float4 against the native coordinate planes and checks unchanged editor source and both generated shader passes.

Compiler support includes static CUDA module declarations, fixed two-dimensional constant arrays up to 256 scalar elements, and named pointers to column zero of constant rows. Row indices are evaluated once. Pointer reassignment, writes, escapes, shadowed aliases and unsupported dimensions remain rejected. Two-dimensional initialized constant arrays remain unsupported. Existing `__umul24` support required no changes.

The original NVIDIA source is BSD-3-Clause. The host pipeline, packing helper, point renderer and tests are MIT project code. See the repository's third-party notices.
