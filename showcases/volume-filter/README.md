# NVIDIA 3D volume filter

[Open voxel filtering](../../sandbox.html?example=volume-filter) · [Nearest voxels](../../sandbox.html?example=volume-filter-nearest) · [Upstream normalized coordinates](../../sandbox.html?example=volume-filter-upstream)

The sandbox compiles NVIDIA's original `d_filter_surface3d` and its original byte conversion helper. CUDA function bodies are unchanged. Device declarations are extracted from `volume.h`, `volumeFilter.h` and `volumeFilter_kernel.cu` at CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. The source and 32 × 32 × 32 `Bucky.raw` data retain NVIDIA's BSD-3-Clause licensing; the host integration and illustration are MIT.

The preview displays grayscale slices of the final GPU texture. Moving the Z slider reads a different slice; it does not rerun the kernel or alter source. CUDA and generated WGSL are available in Compare. This is the filter stage, not the complete desktop application's filter/animation/ray-marching pipeline.

## Why the modes look different

The pinned upstream `volume.cpp` enables normalized coordinates, wrap addressing and linear sampling. Its kernel passes integer voxel indices plus filter offsets to tex3D. In that configuration, integer increments wrap to equivalent positions. Native CUDA and WebGPU both produce a uniform volume for the tested settings.

The default showcase explicitly selects **unnormalized voxel coordinates and clamp addressing** in the host texture configuration. The alternate nearest preset changes only the sampling filter. The upstream preset retains normalized/wrap/linear settings, so its flat output remains visible and reproducible. None of these modes modifies the CUDA kernel.

All three presets use three example host-provided `(x,y,z,weight)` taps: `(-.125,0,0,.25)`, `(0,.25,0,.5)`, `(.125,0,.25,.25)`, with bias `.0625`. These are validation coefficients, not the original desktop UI's animated blur/sharpen presets. Edit the flattened `constant.c_filterData[i].x/y/z/w` launch settings to change them. The displayed image is the first filter pass, at Z=16 by default.

## Validation

Sixteen captured volumes cover synthetic 8 × 8 × 4 and original Bucky 32³ data, four sampler/coefficient configurations, and two sequential filtering passes. Hardware WebGPU matches every native byte: 264,192 voxel comparisons with zero mismatches. Between the two GPU dispatches there is no CPU readback or data upload.

The native harness is `tests/volume-filter-native.cu`. Build with `nvcc -O3 --fmad=false -std=c++17 -arch=native -I.local/nvidia-audit/Common tests/volume-filter-native.cu -o .local/nvidia-checks/volume-filter.exe`; run without arguments for synthetic data and with `bucky` for original data. Native captures are `reports/volume-filter-[bucky-]native-*.bin`. Exact agreement is for these tested inputs and arithmetic settings, not a universal floating-point guarantee or a performance benchmark.

The hardware suite also exercises the generic slice readback and invalid slice bounds. `scripts/test-volume-filter-sandbox.mjs` compares 12 displayed slices (12,288 pixels) against native captures, checks all three modes, unchanged source, no redispatch during slice selection, generated WGSL and mobile layout.

## Representation and scope

Read-only byte volumes use r8unorm. Writable byte volumes use rgba8unorm storage, with the CUDA byte in red and four physical bytes per voxel. Later tex3D calls read red directly on the GPU. The runtime offers normalized coordinates by default and unnormalized coordinates with clamp addressing. Nearest voxel sampling uses integer texture loads to preserve boundary selection.

The sandbox supplies the normal CUDA compilation definition `__CUDACC__=1` when extracting device code. Host allocation, constants and launch settings are supplied by the sandbox; arbitrary CUDA desktop host code is not executed. Other VolumeTypeInfo specializations, the full desktop pipeline and unrelated kernels are not claimed as supported by this showcase.
