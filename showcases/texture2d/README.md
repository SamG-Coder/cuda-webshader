# NVIDIA texture rotation

[Open the texture rotation sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=texture2d).

This runs the unchanged `transformKernel` from NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/0_Introduction/simpleTexture/simpleTexture.cu`. The kernel and original
`teapot512.pgm` retain NVIDIA's BSD-3-Clause terms.

The host decodes the 512×512 binary PGM into normalized float input pixels,
matching NVIDIA's PGM loader. A real `r32float` texture and sampler provide
`tex2D<float>` operations in generated WGSL. All rotation and sampling execute
on the GPU. The generic `gray-f32` preview maps final output values from 0–1
to grayscale; it does not transform the input image in JavaScript.

The default is NVIDIA's angle of 0.5 radians, linear filtering and wrapped
coordinates. Change `theta` in Launch settings and click Compile & Run.
`filter` can be `nearest` or `linear`. Float texture sampling requires the
WebGPU `float32-filterable` feature.

The original kernel has no bounds guard. The preset launches exactly 512×512
threads: 64×64 blocks of 8×8, with 262,144 output records. Keep the dispatch
extent and output dimensions consistent when changing launch settings.

Validation compares every output float and sixteen trailing guards against
native CUDA at angles 0.5, 0 and −0.7 with linear filtering, and at 0.5 with
nearest filtering. Native CUDA also checks an independent double-precision
sampling reference. Linear comparisons allow 0.005 absolute error for texture
interpolation precision; observed WebGPU/native maxima are approximately
0.00164, 0 and 0.00164.

Nearest results match exactly except for five pixels at texel boundaries.
For each differing pixel, the test requires the independently calculated
coordinate to be within 0.001 texel of a boundary and both results to equal
one of the adjacent original texel values. It does not apply a general error
tolerance to nearest filtering. Small trigonometric rounding differences can
choose opposite sides of a discontinuous nearest-sampling boundary.

The built sandbox compares every displayed pixel with native CUDA at the
default angle and after editing the angle to zero. Source/WGSL comparison and
mobile layout are checked too. See `reports/texture2d-native.txt`, the texture
rotation entry in `reports/nvidia-regression-gpu.json`, and
`reports/texture2d-sandbox-check.json`. These verify correctness, not a
CUDA-versus-WebGPU performance advantage or the original desktop host APIs.
