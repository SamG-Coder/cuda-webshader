# NVIDIA cubemap texture: bring-up

Candidate: pinned cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu`.

## Native reference

The original `transformKernel` body is retained verbatim in `tests/cubemap-kernel.cuh`, with NVIDIA's BSD-3-Clause notice. The MIT capture harness reproduces the sample's default 64 by 64 faces, six-face float array, ascending input values, normalized linear sampling, wrap address settings, and 8 by 8 blocks. All 24,576 native results match the original expected negation.

Additional MIT probe code samples 12 directions near the edges of all six faces, with the CUDA texture's seamless flag disabled and enabled. All 12 results change between modes. The original descriptor is zero-initialized, so it uses the disabled mode. These captures are necessary regression references: center-only sampling cannot establish edge-filtering fidelity.

CUDA exposes the flag explicitly in [cudaTextureDesc](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-runtime-api/structcudaTextureDesc.html). Face selection is described in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html#cubemap-textures).

## Compiler, runtime and sandbox

The original `texCubemap<float>` now compiles, including texture propagation through device helpers. The runtime uploads six scalar float layers; generated WGSL selects a face and its coordinates before sampling that layer. The original wrap address mode is preserved. Native probes caught and corrected a clamp-vs-wrap difference that the original face-center lookups did not expose.

All 24,576 original outputs and all 12 nonseamless edge probes match native CUDA exactly. Seamless requests are explicitly rejected. The sandbox adds a labelled MIT helper to arrange faces in a 3 by 2 atlas, and independently checks all original outputs, atlas orientation and rendered pixels. The main-page card links only to the sandbox.

## Reproduction

Build `tests/cubemap-native.cu` with NVCC `-O3 --fmad=false -std=c++17 -arch=native`, adding the pinned `.local/nvidia-audit/Common` include directory. Run the capture executable from the repository root. Input/output files and their SHA-256 hashes are recorded in `reports/cubemap-native-manifest.json`.
