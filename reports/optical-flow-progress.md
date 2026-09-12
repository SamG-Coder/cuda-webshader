# Optical flow: HSOpticalFlow

Current status: the complete native-compared pipeline and standalone sandbox showcase are implemented. See showcases/optical-flow/README.md, optical-flow-check.json and optical-sandbox-check.json.

## Initial compiler milestone

The next candidate is NVIDIA's six-kernel Horn-Schunck optical-flow sample. Target the original 640 × 480 frame10/frame11 image pair, alpha 0.2, five pyramid levels, three warps per level and 500 Jacobi iterations per warp. This is 7,500 Jacobi dispatches in the complete algorithm; the full sample is not yet connected to the sandbox.

All six original device function bodies are extracted without edits in tests/optical-flow-kernels.cuh. Header/host wrappers are excluded, the NVIDIA BSD-3-Clause notice is retained, and the original `template<int bx,int by>` declaration is retained.

Compiler additions:
- Up to four homogeneous integer template parameters on kernels and device helpers; integer kernel arguments are explicit nonnegative decimal 32-bit signed values. Counts, ranges, shadowing and mixed type/integer declarations are checked. Parameter substitution also reaches nested helper arguments and shared array dimensions.
- Plain volatile shared float/int/uint scalars and arrays use atomic loads/stores. Floats use bitcasts to atomic uint storage. Pointer/reference escapes and implicit atomic compound arithmetic remain unsupported. This preserves individual accesses without inventing synchronization; the original cg::sync block barrier remains necessary.

[NVIDIA documents that volatile does not provide inter-thread synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html). [WGSL atomic operations are relaxed; workgroupBarrier provides workgroup acquire/release synchronization](https://www.w3.org/TR/WGSL/#memory-semantics). The mapping supports the race-free, explicitly synchronized solver here and does not promise CUDA warp-lockstep behavior.

Native validation runs the original JacobiIteration body for 500 iterations, capturing U/V at steps 1, 8 and 500. Cases are 67 × 19 with a padded stride of 96 and a 32 × 6 tile, and 128 × 96 with a 16 × 4 tile. All measured values match CUDA exactly at each capture; padding remains zero. Tests retain small float tolerances for cross-backend checks. See optical-jacobi-check.json and optical-jacobi-native.txt.

Remaining integration: image pyramid/downscaling, mirrored float textures, warp and derivative stages, bounded repeated solver dispatches with GPU buffer swaps/clears, coarse-to-fine upscaling, full native flow comparison and a motion-field preview. Do not add a runnable showcase card until that full sequence is working and verified.


## Complete pipeline

The original full configuration now executes through bounded pipeline repeats, GPU buffer clears, texture copies and alternating output buffers. All 614,400 final motion components match original native CUDA output exactly on the tested adapter. The six device function bodies remain unchanged. The showcase displays hue for direction and saturation for displacement, with no intermediate CPU readback. The final pipeline expands to 7,669 GPU operations, including 7,500 Jacobi dispatches.
