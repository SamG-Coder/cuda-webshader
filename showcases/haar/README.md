# NVIDIA Haar wavelet

[Open the Haar sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=haar).

This runs the original `dwtHaar1D` kernel from NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`, in
`cpp/5_Domain_Specific/dwtHaar1D/dwtHaar1D_kernel.cuh`. The kernel body and
BSD-3-Clause notice are retained. Constants come from its desktop host file;
the square-root macro omits the host's redundant trailing semicolon.

The sandbox transforms 4,096 deterministic random scalar samples. Four blocks
of 512 threads each calculate ten decomposition levels. A 16-byte GPU copy
moves the four remaining approximation coefficients into the input prefix.
A second launch with two threads computes the last two levels; a four-byte
GPU copy puts the final approximation at output index zero. The output is the
coarsest approximation followed by detail coefficients from coarse to fine.
The preview reads the final output only; no JavaScript transform or intermediate
CPU transfer participates in execution. Both generated shaders can be inspected.

The generic sandbox supports `copies` on the main launch and each additional
pass. Each copy specifies `source`, `target`, `byteLength`, and optional
`sourceOffset`/`targetOffset` in bytes. Copies execute after their own launch,
before the next pass. Ranges must fit distinct buffers, have matching element
types, and be four-byte aligned. At most eight copies per launch are allowed.
Existing full-buffer feedback still executes after the complete sequence.

Native CUDA and NVIDIA WebGPU both verify all output coefficients for lengths
4, 32, 1,024 and 4,096 against an independent pairwise reference, including
untouched output guard elements. The absolute tolerance is 0.00003. See
`reports/haar-native.txt` and `reports/nvidia-regression-gpu.json`.
The sandbox check additionally verifies the actual preset, two dispatches,
both generated shader choices and mobile layout. These are correctness checks,
not performance measurements.

Launch settings are explicit, not a general signal-length planner. In the
unchanged upstream kernel, `approx_final` is written only when `dlevels > 1`.
A single-level launch therefore cannot supply the final approximation through
this copy sequence. The demonstrated sizes avoid that case; arbitrary lengths
and the desktop application's complete host setup are not claimed as supported.
