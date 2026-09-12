# NVIDIA pre-integrated volume

[Open the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=volume-preintegrated)
or [run locally](http://localhost:5173/sandbox.html?example=volume-preintegrated).

This runs the original transfer integration, layered table generation and
ray-marching functions from NVIDIA CUDA Samples `volumeFiltering`, revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`. The original device function bodies,
scalar default arguments and both nine-colour transfer functions are retained.
The host enum values are supplied as preprocessor constants in the isolated
source; CUDA host allocation and launch calls are represented by `pipeline.json`.

The five GPU dispatches integrate transfer function 1, generate layer 1,
integrate transfer function 0, generate layer 0, and render the Bucky volume.
Each transfer layer is 1024 × 1024 float4 entries. The two layers occupy 32 MiB
and stay on the GPU. Only the final 256 × 256 image is read for display.

The visibly different halves are intentional: the original ray marcher chooses
the transfer layer using `int tfid = (pos.x < 0)`. One layer has mostly green/cyan
colours and lower opacity; the other uses a broader palette. Colours therefore
come from the original CUDA sampling and blending code.

Use **Compare** and the shader selector to inspect generated WGSL for each of
the three unique kernel entries. In **Launch settings & buffer inputs**, the
last pipeline step exposes `density`, `brightness` and the inverse-view matrix.
Click **Compile & run** after editing. Changing its entry to
`d_render_preint_off` selects the original non-pre-integrated lookup mode. This
preview is a ray-marched image, not an orbitable triangle mesh.

Native CUDA and real NVIDIA WebGPU comparisons cover all components of two
32 × 32 transfer tables, plus six 256 × 256 renders: 32² and 1024² transfer sizes,
front and rotated cameras, and the non-pre-integrated mode. Maximum observed
table error is 1.1920929e-7; every rendered channel differs by at most one 8-bit
level. The screenshot and browser checks also verify unchanged editor source,
CUDA/WGSL comparison, mobile layout, resource cleanup on example switching,
and zero intermediate CPU readback.

This showcase covers the rendering portion of `volumeFiltering`. It does not
run the desktop OpenGL UI or insert a volume convolution before rendering.
Double expressions use the compiler's integer emulation; no native FP64 speed
or complete desktop application performance equivalence is claimed.

NVIDIA source and transfer data retain their BSD-3-Clause notice. Compiler,
runtime, launch configuration, icon and validation harness are MIT project code.
See [validation details](../../reports/volume-preintegration-progress.md).
