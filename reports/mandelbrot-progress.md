# Mandelbrot compiler progress

Candidate: NVIDIA CUDA Samples `cpp/5_Domain_Specific/Mandelbrot`, revision `5443602d89ed99aede2e4b7bf329daddeadb320e`. No Mandelbrot showcase card has been added yet.

The first issue was generic desktop extraction: `Mandelbrot_kernel.cuh` has whole device functions inside `#if 1` / `#else` regions. The importer used to strip those surrounding directives, importing both implementations and surfacing an invalid declaration in the inactive branch. The fix retains conditional directives around extracted declarations while omitting unrelated host-only conditional regions. It preserves original line positions and function bodies.

Native CUDA and WebGPU probes now verify both branches of a conditional source, including an intentionally invalid inactive function. Nested conditionals and conditional constant declarations are unit-tested. 412 unit tests and 119 NVIDIA hardware GPU checks pass, along with compile-all and the static build. This validates an importer prerequisite, not the Mandelbrot renderer.

With the original header and CUDA file extracted together, `Mandelbrot0<float>` now reaches the active kernel signature and reports the next blocker: the by-value `const uchar4 colors` parameter. Packed uchar4 buffers and locals already work, but packed kernel parameters do not. Further original-source requirements include the `bool isJ` launch parameter and per-component writes to packed storage pixels in accumulated frames. These need correct ABI and storage semantics before claiming the original renderer works. The secondary anti-alias pass and double-single path require separate validation; no claim of double-precision WebGPU support is made.

Next: implement the necessary generic launch/storage support without changing NVIDIA function bodies, then compare actual rendered images against native CUDA before exposing a sandbox showcase.
