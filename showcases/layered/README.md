# NVIDIA scalar layered texture

[Open in sandbox](../../sandbox.html?example=layered).

Original `transformKernel` from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu`. The CUDA body is unchanged.

The complete original workload uses five 512 by 512 scalar float layers. Each layer contains the same ascending input ramp. Five original launches read normalized texel-center coordinates with linear wrap sampling, negate the value, and add the layer index. All 1,310,720 outputs are retained.

The MIT display helper arranges layers 0 through 4 horizontally. Labels identify the panels; the shared grayscale range is -262143 to 4. The gradients look similar because the original layer offset is only 0 through 4. Neither input nor CUDA computation has been modified to exaggerate that difference.

## Verification

The native capture reproduces the original texture descriptor, input initialization and launch sizes; every result matches the original expected negation plus layer index. All 1,310,720 WebGPU results equal the native values. A separate texture with distinct layer contents checks layer selection and texture propagation through a device helper, so the repeated original input cannot hide selection mistakes.

The sandbox test independently checks every native output, its position in the five-panel atlas, all 1,310,720 rendered pixels, the labels and unchanged displayed source. Both generated shader passes appear in the comparison view.

The compiler now accepts `tex2DLayered<float>` alongside the existing float4 path. The runtime uses `r32float` or `rgba32float` layered storage with matching record widths. Existing float4 callers retain their default format.

Run `node scripts/test-layered.mjs`, and after `npm run build`, `node scripts/test-layered-sandbox.mjs`. Set `CW_BASE_URL` for public deployment checks. Build `tests/layered-native.cu` with NVCC `-O3 --fmad=false -std=c++17 -arch=native` and the pinned Common include directory to reproduce native captures. The capture manifest records file sizes and SHA-256 hashes.

NVIDIA code retains BSD-3-Clause licensing. The separately labelled display helper, capture harness, compiler/runtime changes and pipeline are MIT project code.
