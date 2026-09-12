# NVIDIA pre-integrated volume renderer

Candidate: `cpp/5_Domain_Specific/volumeFiltering/volumeRender_kernel.cu` at
NVIDIA CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The full pre-integrated renderer is **not yet a working sandbox showcase**.
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

The MIT ABI probe matches native CUDA exactly for 64 float4 records (256 float
components), including negative values and values above one. A 37-record partial
launch preserves the remaining initialized texels. Four invalid extent cases
are rejected. Native source and captures are in `tests/float4-surface-native.cu`,
`reports/float4-surface-native.txt` and `reports/float4-surface-native.bin`.

## Remaining work

The original integration kernel runs under NVCC and produces 64 finite float4
records, saved in `reports/volume-transfer-native.bin`. This capture is a native
baseline, **not a WebGPU equivalence result** for that original kernel.

Translation currently stops at `1.0 / float(extent.width - 1)`: the unsuffixed
literal promotes that scalar calculation to double precision. The loop condition
also contains `to + incr * 0.5`. Replacing these literals with float literals
would change the source and its arithmetic; this work does not do that.

After preserving those arithmetic semantics, the original pipeline still needs
layered float4 surface stores, layered transfer texture sampling, and sandbox
pipeline bindings for these resources. The existing volume-filter showcase
continues to represent its validated convolution stage only.

Validation for this foundation: 517 unit tests, 157 real NVIDIA WebGPU checks,
compile-all and the static build pass. No software GPU tests were run, and no
performance claim is made for the unfinished renderer.
