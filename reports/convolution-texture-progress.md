# NVIDIA convolutionTexture: in progress

Target: both original texture-based row and column filters from NVIDIA CUDA
Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, in
`cpp/2_Concepts_and_Techniques/convolutionTexture/`.

The intended showcase runs the original radius-eight, 17-coefficient unrolled
helpers and both kernels, with the intermediate buffer copied into the texture
on the GPU. This is not yet a verified showcase.

## Verified: descending integer helper specializations

The original helper calls `convolutionRow<i - 1>` and terminates at an explicit
`convolutionRow<-1>` specialization. Helper template arguments now accept
bounded decimal signed integer arithmetic with +, -, *, /, %, unary signs and
parentheses. Substitution also works inside nested helper arguments and from
integer kernel parameters. Equivalent integer specialization expressions have
one canonical key, and duplicate definitions are rejected.

Each expression has at most 64 tokens and 32 nesting levels. Arithmetic must
stay within signed 32-bit bounds, with truncating division and signed remainder.
Division by zero and overflow are rejected. At most 128 device-helper template
instances are allowed. Runtime recursion remains unsupported. Negative template
values, including INT_MIN, retain signed semantics in generated WGSL.

Numeric preprocessing now accepts balanced outer parentheses, including the
original `#if (UNROLL_INNER)`. General preprocessor expressions remain unsupported.

The MIT-owned arithmetic probe in `tests/integer-helper-expressions.cu` follows
the same descending-specialization pattern, using GPU buffer inputs. Native CUDA,
the typed CPU test interpreter and NVIDIA WebGPU all return exactly
187, 0, 7, −2, INT_MIN and −3. This probe is compiler validation, not a replacement
for the NVIDIA filter or a showcase. See
`reports/integer-helper-expressions-native.txt`,
`tests/integer-helper-expressions.test.mjs`, and its GPU regression entry.

## Verified: texture helpers and expression macros

The compiler propagates a single sampling format through connected kernel and
helper texture parameters. Each WGSL helper receives the actual texture and
sampler as separate arguments. This covers ordinary and explicitly specialized
integer helpers, including the unused texture argument in the negative base
specialization. Different texture resources and sampler settings can use the
same helper. Mixing formats on one parameter chain, opaque handle returns,
local aliases and implicitly resolved texture helper templates are rejected.

Native CUDA and NVIDIA WebGPU return the same eight exact values for two input
images using distinct nearest/linear samplers, swapped arguments, and descending
helper templates. Separate GPU checks cover 3D volume and float4 transfer handles.
See tests/texture-helpers.cu and reports/texture-helpers-native.txt.

The original IMAD macro is preserved by desktop extraction and expanded as a
parenthesized expression AST. Each parameter occurrence must itself be fully
parenthesized to preserve argument precedence. Bodies are limited to 1,024
characters and 16 parameters; expanded ASTs are limited to 65,536 nodes. Nested
expression macro calls are rejected. Numeric object macros can use bounded
integer arithmetic with previously defined numeric constants, covering the
original KERNEL_LENGTH expression. Native CUDA and GPU probes verify signed
24-bit IMAD, argument precedence, repeated parameters and dependent constants.
See reports/expression-macros-native.txt.

Both original convolutionTexture kernels and their unrolled helper chains now
compile and pass real GPU shader validation, using the licensed fixture in
tests/convolution-texture-kernel.cuh. That check is explicitly compilation-only;
it does not establish correct image results with the current runtime settings.

## Remaining work

- Support unnormalized texture coordinates. The original CPU reference clamps
  boundary indices; compare native boundary behaviour explicitly rather than
  assuming normalized wrapping applies.
- Copy the row output buffer into the texture on the GPU before the column pass.
- Compare complete native and WebGPU results, including borders and odd sizes,
  then add a standalone card linked to the sandbox.

The original CUDA function bodies remain unchanged. No convolutionTexture
showcase card has been added while complete output validation is outstanding.
