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

## Remaining work

With the header constants provided, the imported original source now gets past
integer template expansion and reports `Invalid helper parameter` at the
`cudaTextureObject_t texSrc` helper argument. Required next steps are:

- Pass texture/sampler bindings through ordinary and specialized device helpers.
- Preserve the original IMAD expression macro during desktop-source extraction
  and preprocessing. The compiler already supports the underlying __mul24.
- Support unnormalized texture coordinates. The original CPU reference clamps
  boundary indices; compare native boundary behaviour explicitly rather than
  assuming normalized wrapping applies.
- Copy the row output buffer into the texture on the GPU before the column pass.
- Compare complete native and WebGPU results, including borders and odd sizes,
  then add a standalone card linked to the sandbox.

The current prerequisite tests do not prove that either complete filter pass
runs. The original CUDA function bodies will be preserved.
