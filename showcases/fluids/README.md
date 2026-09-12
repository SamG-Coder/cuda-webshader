# NVIDIA fluidsGL sandbox

[Run the fluid showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fluids).

Drag across the preview to stir 262,144 particles in a 512 × 512 periodic particle domain. The five original NVIDIA device functions are preserved in kernel.cu: addForces_k, advectVelocity_k, diffuseProject_k, updateVelocity_k and advectParticles_k. Their BSD-3-Clause notice remains in the source.

The host pipeline replaces CUDA memory management and cuFFT with WebGPU buffers, pitched float2 texture copies, and this project's MIT real FFT adapter. It runs forward transforms, diffusion/projection and unnormalised inverse transforms entirely on the GPU. The generated comparison view includes all five original kernels and six FFT helper entries.

The preview uses one-pixel green points with alpha 0.5 on black. Host inputs deliberately use deterministic cell-centred particles rather than the original jittered initialization, for reproducible native comparisons. A small central force runs continuously; dragging substitutes pointer-derived force for that frame. These inputs and controls are host configuration, not edits to the original kernels. Dense points can appear as a solid green field at small viewport sizes.

Validation uses native CUDA/cuFFT captures at steps 1, 8, 32 and 64 for both 64² (padded velocity rows) and 512² domains. Comparisons measure velocity maximum/RMS and periodic particle-distance maximum/RMS. Cross-backend floating-point and texture-filtering differences accumulate, so these are tolerance comparisons rather than bitwise claims. Full solver checks require velocity maximum < 0.0002 and RMS < 0.00001, and particle maximum < 0.25 grid cells and RMS < 0.005 grid cells. Padding must remain unchanged; all outputs must be finite. The sandbox test separately checks the 512² preset against step 64, drag input, mobile layout and switching presets, on NVIDIA hardware.

Reports: ../../reports/fluids-solver-check.json and ../../reports/fluids-sandbox-check.json. No software WebGPU adapter is used. No native graphics pixel-equivalence or performance speedup is claimed.
