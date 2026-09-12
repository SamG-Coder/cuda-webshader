# NVIDIA Sobel edges

[Run edge detection](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=sobel)
or [view the original image through SobelCopyImage](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=sobel-copy).
Both links open the sandbox with editable CUDA and generated WGSL comparison.

`kernel.cu` retains the unchanged `ComputeSobel`, `SobelTex` and `SobelCopyImage`
and `SobelShared` functions from NVIDIA CUDA Samples commit `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/SobelFilter/SobelFilter_kernels.cu`. `Pixel` is the original
unsigned-char value alias. `teapot.pgm` is the original 1024 × 1024 input. NVIDIA's
BSD-3-Clause notice remains in the source; the compiler and host setup are MIT.

## Pipeline and preview

The original kernel samples unsigned-byte texture elements using unnormalized,
nearest coordinates. NVIDIA sets wrap on its unnormalized texture; the supported
CUDA behavior here is clamp. The WebGPU resource is an `r8uint` texture, and the
compiler emits explicit point loads with clamped pixel indices. Values remain
integers from 0 to 255, with no normalized texture conversion in the kernel.

The original pitched cast through `char*` is preserved. Byte stores use packed
atomic compare/exchange to avoid overwriting neighbouring lanes' pixels. This
preserves independent byte writes; it does not make conflicting CUDA writes to
the same byte race-free. Wider same-type casts require a provably aligned offset or a validated integer launch pitch factor.

Edit `fScale` in the edge preset's pipeline step (or `fscale` in the copy preset).
The validated scales are 1 and 0.25. The edge kernel casts the gradient sum to
`short` before clamping, exactly as NVIDIA wrote it. Large scale values can exceed
that type's range; the preview does not substitute a different algorithm.

The default launch uses 1024 blocks of 128 lanes: one block per image row, with
lanes looping over columns. `Pitch` is bytes per output row, and must be at least
`w`; the buffer must fit `Pitch * h`. Dispatch exactly `h` blocks in X with one
block in Y/Z. The displayed image uses tightly packed rows, so its preset uses
`Pitch == w`. Input decoding, launch settings and display are host operations;
all image processing comes from compiled CUDA. Final bytes are read for a
standard grayscale canvas preview and displayed with Y flipped, matching the
sample's OpenGL orientation. There are no intermediate image readbacks.

## Validation

Native CUDA and hardware WebGPU agree exactly for both image kernels in four
scenarios: 31 × 17 and 61 × 65 synthetic pitched images, and the original 1024²
teapot at scales 1 and 0.25. All output bytes, row padding and trailing guards
are checked. Native results also match independent stencil/copy equations.
The sandbox test compares every GPU pixel with native CUDA, every displayed
canvas pixel with GPU output, scale edits, the copy preset, unchanged source,
and mobile overflow. Tests use the NVIDIA Blackwell adapter without software
WebGPU. These correctness results are not performance measurements.

Commands: `node scripts/test-sobel-sandbox.mjs` (after `npm run build`),
`node scripts/test-gpu.mjs`, and the native harness in
`tests/sobel-image-native.cu`. See `reports/sobel-image-native.txt` and
`reports/sobel-sandbox-check.json` for captured results.

## Shared-memory path

[Run SobelShared](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=sobel-shared).
The original kernel and its module-scope `extern __shared__ unsigned char
LocalBlock[]` declaration are preserved. It uses a 16 × 4 workgroup, BlockWidth
80, SharedPitch 384, and 2,304 logical shared bytes. Each shared byte uses a
32-bit workgroup slot, so the physical WebGPU allocation is 9,216 bytes and is
checked against the device limit. This is distinct from packed storage buffers.

The original `uchar4*` output pitch must be a multiple of four. The compiler
records that constraint in the launch metadata; invalid values fail before GPU
dispatch. The kernel's scalar w/h and tile dimensions are short values. The
preset covers 1024² pixels using 4 × 256 blocks. Its original `w / 4` bound means
only complete groups of four pixels are produced; use a width divisible by four.
The launch dimensions, SharedPitch and allocation must cover the full tile and
halo. These are the original CUDA kernel's host responsibilities.

Four complete shared-kernel captures match native CUDA and independent Sobel
stencils exactly: 32 × 17, 64 × 65 and the 1024² teapot at scales 1 and 0.25.
Row padding and trailing guards are intact. The shared sandbox test checks both
teapot scales against native and every canvas pixel against the GPU result.
Together with the texture and copy paths, all three image kernels now run.
`FIXED_BLOCKWIDTH` compile-time specialization has not been separately tested.
The desktop OpenGL host program is not executed in the browser.
