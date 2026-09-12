# NVIDIA recursive Gaussian

[Open the image-filter sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=gaussian).

The original `d_recursiveGaussian_rgba`, pixel conversion helpers and
`d_transpose` come from NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`, in
`cpp/5_Domain_Specific/recursiveGaussian/recursiveGaussian_kernel.cuh`.
Their bodies, conditional edge handling and BSD-3-Clause notice are retained.
Desktop includes and the outer include guard are removed. NVIDIA's host source
credits the CImg library, David Tschumperlé and the CImg contributors for the
implementation on which this filter is based.

The sandbox supplies a synthetic 128 × 64 RGBA colour/checker pattern and the
upstream order-zero coefficient formulas with sigma 2. The column filter,
transpose, second column filter and final transpose execute entirely on the
GPU. Two buffers alternate roles; the original image is no longer needed after
the first filter. The first readback occurs after the complete four-pass result.
JavaScript supplies launch parameters and decodes final packed pixels for display;
it does not filter the image. All four generated shaders are selectable.

The frontend additions are reusable: dereferencing a storage pointer (`*p`),
matching float-vector/scalar arithmetic with `+ - * /`, scalar splat constructors,
finite-float `__saturatef`, and chained `=` assignments to named local variables.
Numeric `#if`, `#else` and `#endif` support an integer literal or numeric macro
with optional `!`, including nesting. Full C++ preprocessing, arbitrary operator
overload definitions and general assignments used as expressions remain outside
the supported subset. This implements the used NVIDIA helper-math operations;
it does not execute the included desktop helper library in the browser.

The generic preview accepts `image: {width, height, format: "rgba8"}` for uint
storage with exactly width × height records. Bytes in each uint are R, G, B, A
from least to most significant. A buffer with `fill: "rgba"` and `width` supplies
a synthetic colour pattern. Dimensions, types and allocation sizes are checked.

Native CUDA and NVIDIA WebGPU check every pass against an independent per-channel
recurrence reference at 17 × 9, 32 × 16 and 128 × 64, including output guards.
Odd dimensions exercise partial blocks and transpose tiles. The comparison allows
up to two 8-bit levels per channel for floating-point evaluation and intermediate
RGBA quantization; guard values must match exactly. The sandbox test checks its
own generated input, output pixels, four dispatches, final-only readback and
mobile layout. See `reports/recursive-gaussian-native.txt`,
`reports/nvidia-regression-gpu.json` and
`reports/recursive-gaussian-sandbox-check.json`. These checks are correctness
validation, not a CUDA-versus-WebGPU performance comparison.
