# NVIDIA simpleGL showcase

Open http://localhost:5173/showcases/simplegl/ after `npm start`.

This runs the **unchanged `simple_vbo_kernel`** from NVIDIA's official
[simpleGL CUDA/OpenGL showcase](https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/5_Domain_Specific/simpleGL/simpleGL.cu).
The pinned upstream commit, source path and kernel SHA-256 are in upstream.json.
The kernel function is extracted without edits; its original copyright and
BSD-3-Clause notice are retained. Generated kernel.wgsl is derived from that
BSD-3-Clause source. The project-authored browser and native verification hosts
remain MIT licensed.

The full upstream desktop program is not executed in the browser. Its GLUT
window and CUDA/OpenGL VBO mapping are replaced with a Three.js WebGPU renderer
and the project's shared GPUBuffer bridge. The CUDA function itself goes through
the existing parser and WGSL emitter without compiler changes. No per-frame
position readback occurs; explicit verification reads back the GPU output.

The original calculation generates `(u, sin(4u+t)*cos(4v+t)/2, v, 1)` for every
vertex. The showcase provides 128, 256, 512 and 1024 square grids, orbit/zoom,
pause/resume, and an explicit output verification button. The upstream kernel
has no bounds guard: all exposed grid sizes are exact multiples of its 8×8 block.

## Actual validation on RTX 5080

- NVCC 13.3 compiled native.cu including the same imported kernel; **12/12**
  cases passed at sizes 128², 256², 512² and 1024², times 0, 1.25 and 7.5.
- Real WebGPU execution in Edge passed the same **12/12** mathematical reference
  checks, using absolute and relative tolerance 3e-6.
- Native maximum height error was below 6e-8. Both paths check all four output
  components, including the homogeneous coordinate.
- Browser animation, pause and grid switching passed. The source function hash
  matched the pinned upstream extraction.
- This establishes kernel correctness and a working browser adaptation. It is
  not a benchmark of the original desktop application's OpenGL rendering.

Evidence: reports/simplegl-native.log, reports/simplegl-webgpu.json and
reports/simplegl-showcase.png at the repository root.

## Reproduce

From a Visual Studio x64 developer terminal at the repository root:

```text
nvcc -O3 -std=c++17 -arch=native showcases/simplegl/native.cu -o reports/simplegl-native.exe
reports\simplegl-native.exe
```

With the server running, set CW_CHROMIUM to a WebGPU-capable Chromium executable
if needed, then run `node scripts/test-showcase.mjs`. The test checks the upstream
kernel hash, every grid at three times, rendered startup and pause behavior.
