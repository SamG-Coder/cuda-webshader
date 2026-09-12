# NVIDIA bicubicTexture: complete device pipeline

Target: the original nearest, bilinear, bicubic B-spline, fast bicubic and Catmull-Rom image paths in NVIDIA CUDA Samples revision 5443602d89ed99aede2e4b7bf329daddeadb320e, cpp/5_Domain_Specific/bicubicTexture/bicubicTexture_kernel.cuh. The complete pipeline is verified and has a standalone sandbox showcase.

## Verified prerequisite: multiple helper template types

The original texture filters use separate data and return type parameters, for example tex2DBicubic<T, R>, and forward R into cubicFilter<R> and tex2D<R>. The compiler now accepts up to four explicit built-in type parameters on device helpers. Parameter order, nested forwarding (including reversed argument order), local declarations, casts, references, vector values and type-trait members retain their types. Full explicit specializations are selected using the entire canonical argument list and checked against the instantiated primary signature.

One type parameter can still be deduced as before. Multiple types require explicit arguments; partial deduction, mixed integer/type lists, multiple kernel or trait-struct parameters, default template parameters and variadic templates remain unsupported. Integer helper templates retain their existing bounded arithmetic and negative terminating specializations. Helper instantiation remains limited to 128 instances, and recursive calls, shadowing, mismatched argument counts, unsupported value types and duplicate specializations are rejected.

The MIT validation probe in tests/multiple-helper-types.cu exercises four types, integer/float conversion, full specialization, reversed nested arguments and pixel-coordinate texture sampling through two-type helpers. Native CUDA and hardware NVIDIA WebGPU both produce exactly 3, -30, -27, 7.5, 2, 4.5 from the same input buffers and image. See reports/multiple-helper-types-native.txt and the multiple-helper-types GPU regression entry. Unit tests additionally cover references, traits, vectors, kernel-to-helper forwarding and rejection paths.

## Verified prerequisite: scalar helper default arguments

Primary device helper definitions now support trailing numeric or boolean literal defaults, including an optional numeric sign and numeric macro expansion. Omitted arguments are copied into each call's AST; explicit arguments take precedence. Defaults are inherited from the primary template when a full specialization is selected. Single-type deduction uses supplied arguments only. Overload selection considers supplied argument types and rejects ambiguity when multiple defaulted signatures match.

Defaults are restricted to scalar value parameters. References, pointers, vector defaults, arbitrary default expressions, defaults on kernel entries and defaults declared on explicit specializations remain unsupported. Required parameters cannot follow optional ones. These restrictions are diagnosed instead of silently binding caller variables or changing the supplied argument list.

Native CUDA and NVIDIA WebGPU both produce exactly 2, 4.5, 12.5, 6.5, -2, 2, 5, -10, 9, 3 for tests/helper-defaults.cu. The probe covers omitted and explicit arguments, nested calls, template deduction, inherited specialization defaults, overloads, and texture helpers. See reports/helper-defaults-native.txt and the helper-defaults GPU regression entry.

The parser also recognizes tex2Dgather template-call syntax in an unused helper. This does not implement gather execution: a selected gather call is still rejected. The original unused gather helper parses without rewriting its source.

## Verified prerequisite: packed uchar4 output

The compiler now represents uchar4 as one packed 32-bit word with x, y, z and w in successive low-to-high bytes, preserving CUDA's four-byte storage record. Whole records can be copied through buffers, locals and device helpers. make_uchar4 accepts four scalar components. The uchar and unsigned char scalar spellings are supported for local values and helpers; byte reads preserve their type for overload selection and template deduction, promote to int for arithmetic, and narrow back to eight bits on assignment.

Byte component updates require a named local uchar4. Direct storage/shared component updates are rejected because implementing a byte store as a whole-word read-modify-write could lose another lane's update. Store the complete record instead. Standalone byte buffer pointers and byte scalar kernel uniforms remain unsupported. Static local/shared byte values use 32-bit WGSL storage internally; dynamic shared byte arrays are rejected. Aggregate byte-vector initializers and whole-vector arithmetic are unsupported. Floating-point conversion validation uses values whose truncated integer lies in 0..255; out-of-range or non-finite floating-to-byte conversion is not a portable CUDA contract.

The packed-uchar4 probe validates 257 lanes, 514 four-byte records and 16 output guard records. Native CUDA, the typed CPU reference interpreter, and NVIDIA WebGPU agree exactly on packing, signed/unsigned integer narrowing, finite fractional float truncation, byte wraparound, integer promotion, overload selection, references and a deduced packed helper. See reports/packed-uchar4-native.txt and the packed-byte GPU regression entry.

The built sandbox also passes a pasted uchar4 kernel through compile, dispatch and preview, with all 128 RGBA bytes in its 32-pixel image matching exactly. See reports/packed-image-sandbox-check.json.

All four original render kernels now run in all five modes. Fifteen complete output images match native CUDA and an independent reference within one colour level, including non-power-of-two texture dimensions, image borders, partial blocks and guards. The built sandbox passes all five modes and an edited transform.

The image checks exposed normalized-coordinate rounding at integer texel boundaries. Unnormalized nearest sampling now uses clamped integer texel loads, with the mode propagated through helper calls in hidden uniforms. Bilinear sampling continues through the GPU sampler.

The pinned bicubicTexture PGM has an invalid payload length. Native and WebGPU validation both use the valid attributed simpleTexture teapot image; the source/data distinction is documented in the showcase README.

See [showcase setup and complete validation](../showcases/bicubic-texture/README.md). The original CUDA function bodies remain unchanged.
