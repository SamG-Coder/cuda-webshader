# postProcessGL compiler progress

Candidate: NVIDIA CUDA Samples `cpp/5_Domain_Specific/postProcessGL/postProcessGL.cu`, pinned revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The original kernel has not been added to the showcase yet. Generic desktop extraction preserves its function bodies and the `SMEM(X, Y)` macro.

Completed prerequisites:
- Indexed primary expression macros support reads and lvalue writes through AST substitution. Parameter parentheses and balanced delimiters are required; surrounding textual operators remain rejected.
- `#ifdef` and `#ifndef` recognize numeric definitions (including zero), expression macros, forwarding macros, and externally supplied numeric defines. Existing nested branch state is preserved. This is still bounded preprocessing, not a complete C preprocessor.
- Native CUDA and real NVIDIA WebGPU agree on a shared `uchar4` tile probe using the original SMEM definition: packed result `ff0d0b07`. This is a compiler prerequisite probe, not validation of the complete postProcessGL image algorithm.
- 402 unit tests and 115 NVIDIA GPU checks pass. Compile-all and static build pass.

Next measured blocker: original `getPixel` calls `tex2D<float4>(inTex, x, y)` with integer coordinates. The current compiler only accepts scalar float 2D texture results and float coordinates. Extend texture format inference, helper parameter propagation, runtime RGBA image upload, and coordinate conversion before executing the original kernel. Then validate its halo loading and dynamic shared tile against native CUDA and an independent image reference, implement sandbox host configuration, and add one sandbox-linked card only after complete image checks pass.

Preserve the original CUDA bodies. The default upstream path uses a float4 texture; the optional `USE_TEXTURE_RGBA8UI` branch has a separate unsigned texture requirement. Do not claim desktop OpenGL execution from a compute-only comparison.
