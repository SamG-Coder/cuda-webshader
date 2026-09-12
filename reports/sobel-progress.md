# NVIDIA SobelFilter compiler progress

Candidate source: CUDA Samples commit `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/SobelFilter/SobelFilter_kernels.cu`.

The texture-based image path now runs as a standalone sandbox showcase.
The shared-memory variant now runs too, with four additional exact native image comparisons.
The original `ComputeSobel` device helper is preserved byte-for-byte in
`tests/sobel-compute.cuh`, including its NVIDIA BSD-3-Clause notice.

## Implemented and verified

- Writable packed `unsigned char*` storage uses atomic compare/exchange to
  preserve neighbouring bytes in the same physical 32-bit word. This does not
  make CUDA operations on the same byte atomic: ordinary conflicting writes
  remain a data race. Read-only byte buffers retain stride one and ordinary loads.
- Signed and unsigned short values narrow on assignments and promote in arithmetic.
  Launch parameters use validated 32-bit uniform slots with 16-bit range checks.
  `sizeof(short)` is two and signed integer `abs` is supported.
- Four native byte-store scenarios (1x1, 3x5, 31x9 and 61x65), including pitched
  offsets, helper updates, wraparound and guard bytes, match hardware WebGPU
  exactly over eight dispatches per scenario.
- Four short boundary scenarios match native CUDA for eight values each,
  including signed/unsigned wrapping, promotion, launch parameters and casts.
- The unchanged NVIDIA `ComputeSobel` helper matches native CUDA exactly for
  4,099 neighbourhoods at each scale -1, 0, 0.25, 1 and 4, including 21 guard bytes.
  Native and CPU-oracle tests also check independent stencil equations.
- 498 unit tests and 148 hardware GPU checks pass. The adapter reports NVIDIA
  Blackwell; software adapters were not requested. Compilation of the existing
  catalogue passes.

## Image path completed

`tex2D<unsigned char>` now uses unnormalized point loads from an `r8uint`
texture with clamp addressing. Same-type byte storage supports the original
pitched `char*` round-trip casts. Eight full-image captures from unchanged
`SobelTex` and `SobelCopyImage` match native CUDA exactly, including pitch
padding and guards. The sandbox shows NVIDIA's original 1024² teapot with
editable scale, generated shader comparison and grayscale byte output.
Every displayed pixel was checked against the GPU result.
See `showcases/sobel/README.md` for launch settings and captured evidence.

## Shared path completed

The original `SobelShared` now runs with its module-scope dynamic byte tile,
cooperative group barriers and aligned pitched `uchar4*` output. Four native
captures match all pixels and guards exactly. The sandbox verifies both teapot
scales and all displayed pixels. Logical shared bytes (2,304) and physical
WebGPU workgroup bytes (9,216) are separately accounted for. Invalid pitch
alignment is rejected before dispatch. The complete launch profile and the
untested FIXED_BLOCKWIDTH specialization are documented in the showcase README.

Short storage pointers and dynamic shared short arrays remain explicit
rejections; there is no packed 16-bit storage ABI. General C++/CUDA host
execution is outside the compiler's scope. These are correctness checks, not
performance measurements.
