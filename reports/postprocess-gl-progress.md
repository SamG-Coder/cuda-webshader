# postProcessGL: complete default float4 image path

The original NVIDIA `cudaProcess` and its device helpers now run in the sandbox, with a standalone showcase card linked only to `sandbox.html?example=postprocess`.

Source: CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/postProcessGL/postProcessGL.cu`. Generic extraction preserves the function bodies, conditionals and shared-memory indexing macro. The original `data/teapot_orig.ppm` is copied without byte changes.

Implemented generic requirements:
- Indexed expression macros and definition conditionals (previous commit).
- Inferred float4 2D texture results through helpers, distinct from scalar float images and float4 transfer tables.
- Numeric-to-float texture coordinates, nearest clamped pixel sampling and linear sampling, with per-texture coordinate uniforms.
- RGBA32 float texture uploads and bounded P6 PPM decoding in the sandbox.

Validation: five native CUDA captures versus real NVIDIA WebGPU and an independent reference, using radii 0, 1, 4 and 8. Every RGB channel matches exactly in all cases; output alpha and guard regions pass. The full image is 512 × 512. A separate hardware case checks bilinear RGBA sampling alongside a 1D transfer texture. 406 unit tests, 117 real NVIDIA GPU checks, compile-all, static build and the explorer checks pass. The static sandbox verifies all native image pixels, an edited highlight multiplier, shader comparison and mobile layout.

Scope: the default upstream float4 texture path is complete. The optional `USE_TEXTURE_RGBA8UI` unsigned texture path remains unsupported. These checks run the unchanged compute kernel against the saved upstream input image, not the desktop OpenGL application. They establish correctness, not native-versus-WebGPU performance. See the showcase README for exact launch constraints and reports for measured results.
