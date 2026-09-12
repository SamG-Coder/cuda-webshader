# NVIDIA volume renderer

[Open the volume renderer sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=volume).

The complete original `d_render` ray-marching kernel and its device helpers come
from NVIDIA CUDA Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/5_Domain_Specific/volumeRender/volumeRender_kernel.cu`. Function bodies and
the BSD-3-Clause notice are preserved. The Bucky.raw dataset and nine-entry
colour transfer table also come from that sample.

The browser provides a 32³ normalized 3D texture, a float4 transfer texture,
camera uniforms and a uint output buffer. The original code performs box
intersection, texture sampling, accumulation and RGBA packing on the GPU.
There is no JavaScript simulation or replacement ray marcher.

The default image is 128×128, using 8×8 threads per block and 16×16 blocks.
Launch settings expose density (0.05), brightness (1), transferOffset (0),
transferScale (1), and the twelve `constant.c_invViewMatrix.m[row].component`
values. The default camera is at z=4. Edit settings and click Compile & Run to
recompute. Texture filtering requires WebGPU's float32-filterable feature.

The image preview displays RGB with opaque alpha, matching the desktop sample's
unblended presentation of premultiplied colours. This is a display option;
the kernel's RGBA output remains unchanged.

Native CUDA validation renders 128×128, an odd 65×49 image, and a rotated
128×128 view. The hardware WebGPU test compares every channel against those
native images and checks sixteen trailing buffer guards. Observed maximum
differences are respectively 1, 0 and 1 on an 8-bit channel scale, within a
tolerance of 2. The built sandbox test independently checks every displayed RGB
pixel against the native default view and verifies desktop/mobile layout.

See `tests/volume-render-native.cu`, `reports/volume-render-native.txt`,
`reports/nvidia-regression-gpu.json`, and
`reports/volume-render-sandbox-check.json`. These are correctness checks, not
performance comparisons. The original desktop OpenGL host application is not
executed by the browser.
