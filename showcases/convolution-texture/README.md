# NVIDIA texture convolution

[Open the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=convolution).

Runs both original convolutionTexture kernels and their unrolled helpers from NVIDIA CUDA Samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/2_Concepts_and_Techniques/convolutionTexture/convolutionTexture.cu. The CUDA function bodies, IMAD macro and BSD-3-Clause notice are retained. KERNEL_RADIUS and KERNEL_LENGTH come from the companion header; desktop host code is replaced by sandbox launch settings.

The preset uses the attributed 512 × 512 teapot image from simpleTexture, already stored in showcases/texture2d/teapot512.pgm. Its editable 17 coefficients are the normalized order-16 binomial weights. This input and coefficient selection are showcase settings, not the upstream random-input benchmark.

The row kernel writes a float buffer, a GPU buffer-to-texture copy updates the input texture, and the column kernel writes the final buffer. No intermediate pixels are uploaded or read back through the CPU. The generic grayscale preview reads only the final output. Select either generated shader in the CUDA/WGSL comparison.

Texture settings use normalizedCoords: false, linear filtering and clamp-to-edge. These pixel coordinates match the original native sample, including the effective clamp behaviour of its unnormalized CUDA texture. The runtime supports this mode for float 2D textures with clamp addressing. Hidden coordinate-scale uniforms are passed through texture helpers; user CUDA parameters stay unchanged. Normalized coordinates remain the default for existing examples.

ComputeBatch.copyToTexture supports complete packed float images and aligned source offsets. Rows aligned to 256 bytes use one copy; other widths use one GPU copy per row without CPU repacking. Sandbox textureCopies may follow the root dispatch or an additional pass.

## Validation

Native CUDA and NVIDIA hardware WebGPU each validate all row and column pixels for 128 × 64, 35 × 19, 8 × 5 and 512 × 512 images. Tests include asymmetric coefficients, identity coefficients, the teapot input, clamped borders, partial thread blocks and output guards. Both are also compared with an independent CPU reference used only in tests, with absolute tolerance 0.00003.

See reports/convolution-texture-native.txt, the eight native binary outputs, and the complete convolutionTexture entry in reports/nvidia-regression-gpu.json. The browser test checks every displayed teapot pixel against native output, edits all coefficients to an identity filter and checks the original input, verifies both shader views, rejects an out-of-range texture copy, and checks mobile layout. Results: reports/convolution-texture-sandbox-check.json.

These checks establish image correctness and GPU orchestration, not CUDA-versus-WebGPU performance. They do not execute the original desktop benchmark host in the browser.
