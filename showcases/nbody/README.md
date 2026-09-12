# NVIDIA N-body

[Open the 3D sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=nbody).

The preset runs NVIDIA's original `integrateBodies<float>` kernel and its
`computeBodyAccel`, `bodyBodyInteraction`, reciprocal-square-root, type-trait,
constant and shared-memory helpers. The source excerpts in `kernel.cu` retain
NVIDIA's BSD-3-Clause notices. They come from `bodysystem.h` and
`bodysystemcuda.cu` in `cpp/5_Domain_Specific/nbody` at CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`.

The browser host setup uses 512 synthetic bodies, four blocks of 128 threads,
2,048 dynamic shared bytes, softening squared 0.125, time step 0.002 and damping
0.999. Positions feed the next step through a GPU buffer copy. Velocities update
in place. The preview renders the output buffer directly; no JavaScript force
or integration calculation runs in the preview. Initial positions use the
sandbox's shared sphere-fill option, not NVIDIA's desktop initialization modes.

The compiler option `fullWorkgroups: ['deviceNumBodies']` transforms the matching
linear early-return guard into a workgroup-uniform check. The runtime enforces
that the count is a nonnegative multiple of the block size on every scalar
update, including serialized artifacts. Without this contract, the original
guard remains rejected by WebGPU's uniformity validator. This option supports
one-dimensional blocks and the documented linear guard shape; it does not
disable uniformity validation. The launch must also provide enough position and
velocity records, and `numTiles` must match the source body count divided by the
block size. Arbitrary edited launch settings are not automatically proven safe.

Native CUDA and NVIDIA WebGPU each pass three integration steps against an
independent numerical reference for 32, 96, 128, 128, 384 and 512 bodies using
32- and 128-thread blocks. Checks include positions, velocities, preserved mass
components and allocation guards, with absolute tolerance 0.00003. The two
128-body cases use different block sizes. These are correctness checks, not
performance measurements.

See `reports/nbody-integrate-native.txt` and the N-body integration entry in
`reports/nvidia-regression-gpu.json`. `scripts/test-nbody-sandbox.mjs` checks at
least 120 animated GPU steps, changing rendered pixels, no CPU data uploads or
readbacks during animation, and rejection of a 511-body launch. Its results are
in `reports/nbody-sandbox-check.json`.

The desktop host application, multi-GPU scheduling and CUDA/OpenGL interop are
not ported. This showcase covers the complete single-GPU integration kernel
with a browser host and renderer.
