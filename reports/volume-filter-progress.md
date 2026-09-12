# NVIDIA volumeFiltering compiler progress
Candidate: CUDA Samples `cpp/5_Domain_Specific/volumeFiltering` at revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The original `d_filter_surface3d` reads a 3D texture, applies constant float4 filter weights and writes a byte-valued 3D surface. Its source remains unchanged. The current sample is not running yet and has no showcase card.

## Completed prerequisite: 3D float surfaces
The compiler now infers 2D versus 3D surface bindings from intrinsic use. `surf3Dwrite` accepts float values, global XYZ coordinates, a byte X offset of globalX * 4, and default or explicit trap mode. It rejects mixed dimensions, shifted/unproven coordinates, other formats and other boundary modes. The runtime checks the entire dispatch against the surface dimensions before submission, including Z.

The runtime can allocate or upload r32float 3D storage textures. Existing r8unorm texture uploads retain their format and byte data. This step does not add float-3D sampling bindings or byte surface emulation.

An MIT probe executes the same CUDA function on native NVIDIA CUDA and hardware WebGPU: all 256 float voxels agree across eight workgroups. Hardware checks also reject oversized X, Y and Z dispatches. Native build: `nvcc -O3 -std=c++17 -arch=native tests/surface3d-native.cu -o .local/nvidia-checks/surface3d.exe`. Native result: `surface3d-native.txt`. GPU result: `nvidia-regression-gpu.json`.

## Remaining original-sample requirements
- Import required device declarations without bringing in unrelated host-only Volume/cudaArray structures.
- Support the cudaExtent launch argument with explicit checked host transport and appropriate size_t semantics.
- Compile VolumeTypeInfo<unsigned char>::convert, including its original literal/cast arithmetic.
- Support sizeof(VolumeType) with correct CUDA type semantics.
- Support byte-valued 3D surface writes and sampling the resulting volume without CPU round trips.
- Compare original filtering passes and final preview with native CUDA before adding a standalone sandbox showcase.

The float probe verifies a compiler/runtime prerequisite. It is not evidence that the original byte filter or complete desktop application works.
