# NVIDIA box filter

[Open the colour blur in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=box-filter).

This runs the original `d_boxfilter_rgba_x` and `d_boxfilter_rgba_y` kernels from
NVIDIA CUDA Samples commit `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/boxFilter/boxFilter_kernel.cu`.
`kernel.cu` retains every device/kernel function body and NVIDIA's BSD-3-Clause
notice; desktop includes, host functions and host declarations are omitted.
The original `teapot1024.ppm` image is copied unchanged from that sample.

The row pass uses a sliding window over a colour texture. The column pass reads
the packed intermediate pixels from a GPU buffer. The original sample packs to
8-bit RGBA after each pass, including the intermediate image. Neither pass is
implemented in JavaScript. The final image is read once for display in a canvas.
Its rows are flipped for presentation; this does not change the filter output.

The default is a 1024 × 1024 image with radius 14 and one two-pass iteration.
Edit `r` in **both** steps under Launch settings, then compile and run, to change
the blur radius. Radius 22 is also tested on the complete original image. Input
RGB bytes are unchanged; the PPM loader supplies opaque alpha. The texture uses
linear sampling, pixel coordinates and clamp-to-edge addressing. These are
explicit host settings. Integer texture coordinates still have CUDA's half-texel
sampling behaviour; even radius zero is not necessarily an unchanged copy.

The column kernel has no bounds guard. Its launch must cover exactly `w`
columns, and the original edge loops require `0 <= r` and `2*r < h`. The preset
uses 64 threads per block and widths divisible by 64. The generic sandbox checks
resource budgets and types, but does not prove arbitrary source indexing safe.
The original desktop program, interactive OpenGL loop and repeated-iteration
host orchestration are not executed by this preset.

Compiler support added for this sample:

- Same-allocation pointer rebasing, including `id = &id[x]`, pointer aliases
  and helper parameters. The buffer binding stays fixed and each lane updates
  its own integer offset. Switching allocations, incompatible element types,
  null pointers and floating-point offsets remain rejected.
- Integer/byte scalars promote to float in float-vector arithmetic, matching
  the CUDA helper overloads used by `rgbaIntToFloat(id[0]) * r`. Mixed vector
  element types remain unsupported.

The CPU reference also gives each lane its own pointer-parameter cell while
sharing the underlying data. It is a test oracle, never a browser fallback.

The reusable sandbox pipeline now supports RGBA image previews and PPM texture
inputs as well as triangle meshes. Its JSON declares resources, bindings and
dispatches. The intermediate image stays on the GPU; no intermediate data or
control values are read between the two colour passes. All generated passes
are available in the CUDA/WGSL comparison.

Native validation covers four colour cases: 64 × 37 with radii 0 and 3, plus
the original 1024² image with radii 14 and 22. Both row and final images are
compared (eight complete captures); every channel agrees within one 8-bit
level, and guard records remain intact. This is tolerance-based agreement,
not a bit-exact claim. The native harness disables multiply-add fusion.
Four additional 64 × 32 scalar captures cover the global-memory and texture
versions of both row and column filtering, with float tolerance 1e-6.

`tests/box-filter-native.cu` generates the native captures.
`tests/box-filter-gpu.js` checks all six original kernel entries on real NVIDIA
WebGPU hardware. After `npm run build`, run
`node scripts/test-box-filter-sandbox.mjs` for the built-page test: default and
edited radius, native pixel comparison, displayed pixels, source preservation,
shader comparison, invalid dimensions and mobile layout. Results are in
`reports/box-filter-sandbox-check.json` and `reports/nvidia-regression-gpu.json`.
These are correctness checks, not CUDA-versus-WebGPU performance measurements.
