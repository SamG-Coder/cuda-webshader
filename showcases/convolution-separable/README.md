# NVIDIA separable convolution

[Open the two-pass sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=separable).

The source retains NVIDIA's original `convolutionRowsKernel` and
`convolutionColumnsKernel` bodies from CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`, in
`cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu`.
The radius-eight and length-17 constants come from its common header. NVIDIA's
BSD-3-Clause notice is retained; desktop host functions are not included.

The sandbox initializes a deterministic random 128 × 64 image. The original row
pass writes the intermediate image, then the original column pass writes the
final image back to the first buffer. No CPU readback or image processing occurs
between passes. The numeric preview inspects the final output. The generated
shader selector exposes the WGSL for both passes beside the same CUDA source.

The compiler now supports `+=` and `-=` integer offsets on storage-buffer pointer
parameters and local aliases, including by-value helper parameters. Pointer
origins can temporarily lie before the allocation for halo indexing, as in the
original NVIDIA code, but every dereferenced element must be inside the buffer.
Offsets use 32-bit signed indexing. General pointer reassignment, pointer casts,
pointer-to-pointer arguments and shared/local-array pointer parameters remain
unsupported. Const pointee protections still apply after a pointer moves.

Additional sandbox `passes` specify `entry`, `block`, `groups`, optional scalar
overrides, and `bindings` mapping kernel parameters to existing buffer names.
At most eight additional passes are supported. Passes run in order within one
GPU batch; optional feedback copies happen after the sequence. Every pass is
compiled and validated, and buffer types and launch shapes are checked.

Native CUDA and hardware WebGPU independently verify the row result and the
complete row/column result against a direct convolution reference. The cases
are 128 × 64 with pitch 128 and 256 × 128 with pitch 272, including padding and
end guards. Coefficients are asymmetric to detect reversal mistakes. The
absolute tolerance is 0.00003. See `reports/convolution-separable-native.txt`
and the separable-convolution check in `reports/nvidia-regression-gpu.json`.

`scripts/test-separable-sandbox.mjs` verifies the two hardware dispatches,
preview output against the reference, both shader comparison choices, and
mobile layout. These checks do not constitute a performance benchmark.
