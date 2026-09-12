# NV12 resize and colour conversion: native investigation

Candidate: NVIDIA `NV12toBGRandResize`, revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`.

Status: **not ready for a showcase**. Both original processing orders execute
on the NVIDIA Blackwell GPU with the original default dimensions and batch:
1920 x 1080 input, 640 x 480 output, 24 frames, 20 timing repetitions.
No original CUDA kernel or helper body was changed.

`tests/nv12-capture.cpp` replaces only the sample's file-output utility. It
writes binary planar floats, including all 24 frames. The upstream utility
opens its float output in text mode, which is unsuitable on Windows.
`scripts/capture-nv12.mjs` verifies capture sizes and finite colour ranges,
hashes every frame, and saves each distinct frame plus a complete frame map.
The manifest also records normalized source hashes and current compiler errors.
These captures document observed behaviour; they are **not correctness oracles**.

## Original batch indexing defect

In `nv12_resize.cu`, `resizeNV12BatchKernel` initializes its destination pointer
to `pDstNv12 + px + py * nDstPitch`. Its loop starts at `blockIdx.z`, but this
initial pointer does not include a batch offset. The wrapper launches 24 blocks
in Z for the default batch. Each therefore enters once and writes frame zero;
frames 1 through 23 remain unwritten. The following conversion kernel reads
those unwritten frames. Advancing `p` at the end of the loop does not fix this.

Compute Sanitizer's `initcheck` confirmed uninitialized global memory reads
inside `nv12ToBGRplanarBatchKernel`, including block Z = 1. It reported 61,632
errors before the instrumented application terminated with
`cudaErrorLaunchTimeout` during capture. This was a partial sanitizer run,
not a completed validation. The ordinary native run completed successfully.
The diagnostic excerpt is in `nv12-initcheck-summary.txt`.

The other processing order also produces different frames at the starts of
the two texture tiles (frames 0 and 12). Its linear sampling at tile boundaries
needs separate verification; do not assume repeated input means identical
output or replace the full batch with a single-frame test.

## Compiler and runtime work

- Fixed acceptance of `static` after `__device__` and `__global__`, including
  the original `__forceinline__ __device__ static` ordering. A regression test
  checks that these legal orderings generate identical WGSL.
- Conversion currently stops at reinterpretation between packed byte/vector
  pointers and `uint32_t` pointers.
- NV12 resize requires `uchar2`, packed pair stores and byte-pair texture reads.
- BGR resize currently stops at a local pointer initialized to `NULL`.
- The full-resolution BGR intermediate is 597,196,800 bytes. The original BGR
  wrapper splits its texture into two 12-frame tiles, each still 298,598,400
  bytes, larger than this GPU's 268,435,456-byte WebGPU storage binding limit.
  Preserve the 24-frame workload while addressing resource limits; do not
  silently shrink the candidate to one frame.

## Reproduction

Build the original `bgr_resize.cu`, `nv12_resize.cu`,
`nv12_to_bgr_planar.cu`, and `resize_convert_main.cpp` together with
`tests/nv12-capture.cpp`, using NVCC `-O3 -std=c++17 -arch=native`,
the sample Common include directory, and MSVC `/Zc:preprocessor`.
Keep the pinned upstream checkout in `.local/nvidia-audit`.

Run the executable from the repository root with arguments:

```
-input=.local/nvidia-audit/cpp/5_Domain_Specific/NV12toBGRandResize/data/test1920x1080.nv12 -width=1920 -height=1080 -dst_width=640 -dst_height=480 -batch=24
```

Use `cmd /d /c` on Windows to preserve the sample's `-name=value` arguments.
Then run `node scripts/capture-nv12.mjs`. Native full-batch files remain in
`.local`; the compact, deduplicated captures and manifest are under `reports`.
