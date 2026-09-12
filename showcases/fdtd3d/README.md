# NVIDIA FDTD3d

[Open the 3D volume sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fdtd).

This showcase runs the original `FiniteDifferencesKernel` from NVIDIA CUDA
Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, in
`cpp/5_Domain_Specific/FDTD3d/inc/FDTD3dGPUKernel.cuh`. `kernel.cu` retains the
kernel body and NVIDIA's BSD-3-Clause notice. The two launch constants are copied
from `FDTD3dGPU.h`; desktop includes are replaced by those declarations.

The kernel applies a radius-four, 25-point stencil to a 3D scalar field. It uses
a shared XY tile with halos, per-thread front/back slice arrays, and a constant
coefficient array. The sandbox uses a 32 × 16 × 16 interior volume, four halo
cells on every side, and 32 × 4 threads per block. Both input and output start
with the same deterministic random field, so untouched boundary cells persist.
After each update, a GPU copy feeds the output into the next step.

The generic volume preview maps interior cell indices to XYZ coordinates and
reads their colors directly from the scalar GPU storage buffer. It does not
calculate the stencil in JavaScript. The coefficient preset demonstrates a
smoothing update; it is not a calibrated electromagnetic model.

Constant array elements are editable through scalar keys such as
`constant.stencil[0]` through `constant.stencil[4]`. The compiler supports
one-dimensional constant arrays of 1–256 float/int/unsigned elements, numeric
literal initializers, partial/zero initialization, dynamic indexing, and
per-dispatch snapshots. Each element is packed as a uniform scalar, avoiding
an incompatible CUDA/WGSL array stride. Constant arrays remain read-only.

Native CUDA and NVIDIA WebGPU each pass three updates against an independent
reference for 32 × 4 × 3, 35 × 7 × 5, 64 × 8 × 9, and 32 × 16 × 16 interiors.
The comparison covers every interior and halo element, with absolute tolerance
0.00003. The odd dimensions test partial blocks. See
`reports/fdtd3d-native.txt` and the FDTD3d entry in
`reports/nvidia-regression-gpu.json`.

`scripts/test-fdtd3d-sandbox.mjs` checks sustained GPU animation, changing
rendered pixels, no CPU simulation-data uploads or readbacks during animation,
and rejection of an inconsistent volume shape. Its results are recorded in
`reports/fdtd3d-sandbox-check.json`. These are correctness/interop checks, not
CUDA-versus-WebGPU performance measurements.
