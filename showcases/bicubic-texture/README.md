# NVIDIA bicubic texture filtering

[Open the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=bicubic).

Runs the original device functions from NVIDIA CUDA Samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/5_Domain_Specific/bicubicTexture/bicubicTexture_kernel.cuh. Function bodies and the BSD-3-Clause notice are retained. The generic importer excludes desktop declarations, include guards and host-only qualifiers. The original unused gather template is retained; executing texture gather is not supported and is not needed by these render paths.

## Filter modes

The example menu groups all five modes under NVIDIA bicubicTexture:

| Mode | Original kernel | Sampling |
| --- | --- | --- |
| Nearest | d_render | Point |
| Bilinear | d_render | Linear |
| Bicubic B-spline | d_renderBicubic | 16 point samples |
| Fast bicubic | d_renderFastBicubic | 4 bilinear samples |
| Catmull–Rom | d_renderCatRom | 16 point samples |

Each menu choice loads the same original CUDA source and selects its entry and sampler settings. The default is a 512 × 512 B-spline preview at scale 0.35. In Launch settings, scale changes zoom, tx/ty pan, and cx/cy set the transform centre. CUDA and generated WGSL remain available for comparison.

The input is NVIDIA's valid simpleTexture teapot512.pgm, reused from the existing attributed texture-rotation showcase. The bicubicTexture copy at the pinned revision is malformed: it declares 512 × 512 byte pixels but contains 462,573 payload bytes instead of 262,144. It fails the strict PGM decoder, so it is not loaded or silently truncated. Both the native CUDA validation host and the sandbox use the same valid simpleTexture image. Normalized byte values are uploaded as an equivalent float texture. The original kernels output uchar4 records, packed into exactly four bytes each. Their alpha byte is zero; the generic image preview displays RGB opaquely, matching the desktop sample's visible RGB image.

Unnormalized nearest sampling uses floor and clamped integer texel loads. This avoids selecting the preceding texel when normalization rounds an exact integer coordinate slightly down on non-power-of-two dimensions. Bilinear sampling uses a real GPU sampler. Per-resource mode and coordinate scales are hidden uniforms passed through nested texture helpers; the CUDA code and its parameter list stay unchanged.

## Validation

Three cases run in each of the five modes: a 16 × 12 synthetic texture rendered to 35 × 19, the teapot rendered to 128 × 96 with an offset/scaled view extending beyond its edges, and the 512 × 512 showcase zoom. Native CUDA reads an unsigned-byte normalized texture; WebGPU reads the equivalent float texture.

All pixels are compared against native CUDA and an independent CPU interpolation reference used only in tests. The observed maximum difference is one 8-bit colour level for GPU versus native, GPU versus reference, and native versus reference; the test tolerance is two levels for filtering and quantization differences. Output channel packing, zero source alpha, partial blocks and guard records are checked. See reports/bicubic-texture-native.txt, its 15 binary outputs, and the complete bicubicTexture entry in reports/nvidia-regression-gpu.json.

The static sandbox test switches through all five modes, compares every displayed RGB pixel with native output, verifies opaque display alpha, edits the transform and dimensions, and checks the shader view and mobile layout. See reports/bicubic-texture-sandbox-check.json and scripts/test-bicubic-texture-sandbox.mjs.

These are correctness checks, not CUDA-versus-WebGPU performance measurements. Catmull–Rom can overshoot the input range; the original kernel's byte conversion is preserved. Out-of-range floating-to-byte behaviour is not a portable CUDA contract, so arbitrary images and transforms should not be assumed to have identical overflow behaviour across devices.
