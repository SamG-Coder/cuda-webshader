# NVIDIA pre-integrated volume renderer

Candidate: `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` at
NVIDIA CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The pre-integrated renderer is now a
[working sandbox showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=volume-preintegrated).
The original `d_integrate_trapezoidal` body is retained unchanged in
`tests/volume-transfer-kernel.cuh`, including its double literals.

## Completed support

- `surf1Dwrite(float4, surface, globalX * sizeof(float4))`, with the default
  or explicit trap boundary mode, uses a one-row `rgba32float` storage texture.
- The runtime rejects launches outside that texture, multiple rows/depth lanes,
  and resources with more than one row. The compiler rejects shifted, incorrectly
  scaled or mutated coordinates and mixed surface dimensions.
- `tex1D<float4>` accepts scalar numeric coordinates with CUDA's float conversion,
  including the original kernel's integer literal zero.
- Float4 surface data can be sampled by a subsequent dispatch without CPU transfer.
- Unsuffixed double literals, mixed scalar addition/subtraction/multiplication/
  division, comparisons, negation and conversion back to float preserve binary64
  expression precision through integer-limb WGSL helpers. Explicit double storage,
  double-to-integer casts and double transcendental functions remain unsupported.
- `cudaExtent` size addition/subtraction uses unsigned 64-bit wraparound. Float
  conversion rounds the full integer directly, without first truncating to 32 bits.

The MIT ABI probe matches native CUDA exactly for 64 float4 records (256 float
components), including negative values and values above one. A 37-record partial
launch preserves the remaining initialized texels. Four invalid extent cases
are rejected. Native source and captures are in `tests/float4-surface-native.cu`,
`reports/float4-surface-native.txt` and `reports/float4-surface-native.bin`.

## Unchanged integration kernel verified

The original integration kernel now compiles and runs on real NVIDIA WebGPU.
All components match native CUDA exactly for widths 37, 64 and 1024 (the original
integration width). The 37-record launch also preserves the unused texels in its
64-record allocation. Captures are `reports/volume-transfer-native.bin`,
`reports/volume-transfer-37-native.bin` and `reports/volume-transfer-1024-native.bin`.

The original `1.0 / float(extent.width - 1)` and `to + incr * 0.5` expressions are
retained. Their double arithmetic is performed with integer limbs, including
rounding to nearest with ties to even, rather than changing the CUDA literals.
The transfer input in these checks is a controlled 64-colour table containing
negative and above-one components; this is not yet the complete renderer setup.

Independent arithmetic validation compares 8,817 raw binary64 input pairs with
native CUDA: 88,170 arithmetic/comparison/conversion results plus 52,902 results
from compiler-generated mixed-precision expressions. Bits agree, allowing only
NaN payload/sign differences when both results are NaN. Cases include signed
zero, subnormals, infinities, overflow, cancellation, halfway rounding and seeded
random bit patterns. See `tests/float64-native.cu`, `tests/float64-gpu.js` and
`reports/float64-check.json`.

## Layered renderer completed

`surf2DLayeredwrite(float4)` now uses checked global XY byte offsets and a uniform
or literal layer. The runtime rejects out-of-range layers, over-wide/over-high
dispatches, extra Z lanes and non-writable resources. `tex2DLayered<float4>` passes
through typed helper chains and samples normalized coordinates within each layer.
Layered resources use WebGPU `2d-array` views and `rgba32float` texels.

The original scalar default arguments are preserved in the launch ABI. The
original float overload of `exp` is supported; double transcendental functions
remain unsupported. Five sandbox dispatches create the two original transfer
tables, then ray march the original Bucky volume. Pipeline float4 textures support
storage bindings, and the bounded texture budget is 64 MiB to accommodate the
original pair of 1024² tables (32 MiB).

All 8,192 float components in the 32² two-layer table were compared with native
CUDA; maximum error is 1.1920928955078125e-7. Six images compare 32² and 1024²
transfer tables with the front camera, rotated camera, and pre-integration off.
Each contains 65,536 pixels. Maximum error is one 8-bit channel level, with only
4–7 channels differing per image. The sandbox uses the original 1024² tables
and produces the same front-camera result within one channel level, with no
intermediate CPU readback. See `reports/volume-preintegrated-check.json` and
`reports/volume-preintegrated-sandbox-check.json`.

The unchanged function bodies and transfer initializers are isolated from the
desktop includes, globals and host API code. The rendering pipeline is complete
for this showcase; the separate volume-filter showcase continues to represent
the convolution stage. The desktop OpenGL UI is not translated.

Validation: 526 unit tests, 160 real NVIDIA WebGPU checks, 81 sandbox presets,
compile-all and the static build pass. No software GPU tests were run.
The integer double implementation executes on the real GPU but is not native
hardware FP64; no performance parity is claimed.
