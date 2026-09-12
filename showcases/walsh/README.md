# NVIDIA complete Fast Walsh Transform convolution

[Open in sandbox](../../sandbox.html?example=walsh).

The complete default workload from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/fastWalshTransform`: 8,388,608 data values and 128 kernel values, initialized by the original Windows host sequence with `srand(2007)`.

The three original device functions are unchanged. Each complete transform runs six global radix-4 passes and the shared-memory finish, with 2048 values and 8192 bytes of shared memory per tile. Explicit buffer aliases preserve the original in-place launches. The pipeline transforms data and the zero-padded kernel, runs the original modulation and normalization, then transforms data again. A GPU buffer copy initializes the 128 kernel values.

The 4096 by 2048 preview contains every output value in original order. Its grayscale range of 20 to 45 is a display setting; it does not modify computation. The shader comparison shows the actual compiled pipeline variants, including the in-place storage bindings.

## Validation

The native harness includes the original GPU wrapper and links the original straightforward CPU dyadic convolution. At full size, native versus CPU relative L2 error is 1.56783294e-07, passing NVIDIA's original 1e-6 threshold. Maximum absolute error is 2.67028809e-05.

All 8,388,608 WebGPU outputs match the native float32 bits on the tested NVIDIA GPU. The independent sandbox test verifies the native output SHA-256, all 8,388,608 displayed pixels, unchanged displayed CUDA source, and actual in-place shader in the comparison tab.

Run `node scripts/test-walsh-full.mjs` for the complete native-reference GPU comparison. After `npm run build`, run `node scripts/test-walsh-sandbox.mjs`; set `CW_BASE_URL` to test the published site.

Native capture: build `tests/walsh-full-native.cu` with NVCC `-O3 --fmad=false -std=c++17 -arch=native`, the pinned Common include directory and original `fastWalshTransform_gold.cpp`. The harness writes input, kernel and native output files under `reports/walsh-full-*`; their SHA-256 hashes are in `walsh-full-native-manifest.json`.

NVIDIA source retains BSD-3-Clause licensing. Project pipeline, capture and presentation code are MIT.
