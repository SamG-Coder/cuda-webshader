# NVIDIA bicubicTexture: in progress

Target: the original nearest, bilinear, bicubic B-spline, fast bicubic and Catmull-Rom image paths in NVIDIA CUDA Samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/5_Domain_Specific/bicubicTexture/bicubicTexture_kernel.cuh. The full device pipeline must be verified before adding a showcase card.

## Verified prerequisite: multiple helper template types

The original texture filters use separate data and return type parameters, for example tex2DBicubic<T, R>, and forward R into cubicFilter<R> and tex2D<R>. The compiler now accepts up to four explicit built-in type parameters on device helpers. Parameter order, nested forwarding (including reversed argument order), local declarations, casts, references, vector values and type-trait members retain their types. Full explicit specializations are selected using the entire canonical argument list and checked against the instantiated primary signature.

One type parameter can still be deduced as before. Multiple types require explicit arguments; partial deduction, mixed integer/type lists, multiple kernel or trait-struct parameters, default template parameters and variadic templates remain unsupported. Integer helper templates retain their existing bounded arithmetic and negative terminating specializations. Helper instantiation remains limited to 128 instances, and recursive calls, shadowing, mismatched argument counts, unsupported value types and duplicate specializations are rejected.

The MIT validation probe in tests/multiple-helper-types.cu exercises four types, integer/float conversion, full specialization, reversed nested arguments and pixel-coordinate texture sampling through two-type helpers. Native CUDA and hardware NVIDIA WebGPU both produce exactly 3, -30, -27, 7.5, 2, 4.5 from the same input buffers and image. See reports/multiple-helper-types-native.txt and the multiple-helper-types GPU regression entry. Unit tests additionally cover references, traits, vectors, kernel-to-helper forwarding and rejection paths.

## Remaining work

Compiling the original extracted header currently stops at the default comp = 0 parameter on tex2DBilinearGather. The sample also needs packed uchar4 output and make_uchar4 conversion, the uchar template argument, and matching byte-image texture semantics. The gather helper is not called by the five upstream render paths; its presence must be handled without rewriting the user's CUDA source.

After those compiler/runtime requirements, validate every original render mode against native CUDA and an independent reference, including fractional coordinates, edges and partial blocks. Add the sandbox input/settings/preview and its own card only after full output validation. This report and probe do not claim that the bicubic sample already runs.
