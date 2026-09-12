# Mandelbrot compiler progress

Candidate: NVIDIA CUDA Samples `cpp/5_Domain_Specific/Mandelbrot`, revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. No Mandelbrot showcase card has been added yet.

The first issue was generic desktop extraction: `Mandelbrot_kernel.cuh` has whole device functions inside `#if 1` / `#else` regions. The importer used to strip those surrounding directives, importing both implementations and surfacing an invalid declaration in the inactive branch. The fix retains conditional directives around extracted declarations while omitting unrelated host-only conditional regions. It preserves original line positions and function bodies.

Native CUDA and WebGPU probes now verify both branches of a conditional source, including an intentionally invalid inactive function. Nested conditionals and conditional constant declarations are unit-tested. 412 unit tests and 119 NVIDIA hardware GPU checks pass, along with compile-all and the static build. This validates an importer prerequisite, not the Mandelbrot renderer.

The by-value packed uchar4 and Boolean launch parameters now work. Packed values use one unsigned 32-bit uniform with CUDA x/y/z/w bytes in low-to-high order. Boolean parameters use an explicit u32 transport and a WGSL Boolean comparison; the host accepts true/false or 0/1. Invalid inputs reject the complete update transactionally. Native CUDA and a hardware WebGPU batch agree on four launches, including high-bit colours, byte reversal, Boolean helper arguments and distinct uniform snapshots. 415 unit tests and 120 NVIDIA GPU checks pass; compile-all and the static build pass.

The arithmetic prerequisites now support named local/reference assignment chains, prefix/postfix updates in value expressions and short-circuit loop conditions, const uchar4 references, and scalar type-constructor substitution in templates. An explicitly cast decimal literal, such as T(4.0) with T=float, is folded to float32; arbitrary double expressions and double value types remain unsupported. Updates inside assignment destinations remain rejected because their evaluation ordering requires additional handling.

Native CUDA and WebGPU agree on all 13 numeric values in the reference-effects probe. The native/CPU signed-zero result retains a negative sign; the tested WebGPU backend returned positive zero. The GPU report records negativeZeroSignPreserved=false rather than claiming bitwise equality. 419 unit tests and 121 NVIDIA GPU checks pass; compile-all and the static build pass.

The original combined Mandelbrot header and CUDA file now gets through these arithmetic helpers and reaches the actual packed storage component writes in Mandelbrot0<float>, for example dst[pixel].x during frame accumulation. Implement correct packed-byte storage semantics before rendering and comparing original frames. Const references to storage records also need handling for the secondary anti-alias kernel. Neither an original Mandelbrot image nor its double-single/secondary path has yet been validated.

Next: implement and verify the packed storage requirements without changing NVIDIA function bodies, then compare rendered images against native CUDA before exposing a sandbox showcase.
