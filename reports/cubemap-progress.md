# NVIDIA cubemap texture: bring-up

Candidate: pinned cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu`.

## Native reference

The original `transformKernel` body is retained verbatim in `tests/cubemap-kernel.cuh`, with NVIDIA's BSD-3-Clause notice. The MIT capture harness reproduces the sample's default 64 by 64 faces, six-face float array, ascending input values, normalized linear sampling, wrap address settings, and 8 by 8 blocks. All 24,576 native results match the original expected negation.

Additional MIT probe code samples 12 directions near the edges of all six faces, with the CUDA texture's seamless flag disabled and enabled. All 12 results change between modes. The original descriptor is zero-initialized, so it uses the disabled mode. These captures are necessary regression references: center-only sampling cannot establish edge-filtering fidelity.

CUDA exposes the flag explicitly in [cudaTextureDesc](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-runtime-api/structcudaTextureDesc.html). Face selection is described in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html#cubemap-textures).

## Remaining work

The original kernel currently fails parsing at `texCubemap<float>(tex, cx, cy, cz)`. The frontend needs cubemap intrinsic recognition and texture-type inference. The backend and runtime need a representation that preserves CUDA face orientation, scalar float results and the original nonseamless edge behavior. A direct cube-sampler mapping must not be assumed equivalent without the edge comparisons.

After implementing it, run the unchanged full kernel against all native outputs and the explicit edge probes, then add a six-face sandbox preview and a separate main-page card. No WebGPU success or runnable showcase is claimed yet.

## Reproduction

Build `tests/cubemap-native.cu` with NVCC `-O3 --fmad=false -std=c++17 -arch=native`, adding the pinned `.local/nvidia-audit/Common` include directory. Run the capture executable from the repository root. Input/output files and their SHA-256 hashes are recorded in `reports/cubemap-native-manifest.json`.
