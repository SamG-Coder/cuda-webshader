# DXT texture compression candidate

Selected NVIDIA cuda-samples dxtc at pinned revision 5443602d89ed99aede2e4b7bf329daddeadb320e. Target the complete original 512-square teapot compression, all 16,384 4x4 DXT1 blocks, native comparison and a standalone sandbox showcase with decoded compressed-image preview. This candidate is not yet a runnable browser compressor.

## Original native baseline

Compiled and ran the unchanged dxtc.cu program on NVIDIA Blackwell using nvcc -O3 --fmad=false -std=c++17 -arch=native, the original Common headers and original teapot512_std.ppm / teapot512_ref.dds inputs. It reports Test passed with RMS(reference,result)=0.007950 against the original 0.02 threshold. See dxt-native.txt and dxt-native.dds. The program writes a DDS output beside its input, so it ran against copies in .local/dxtc-run/data and left the pinned source assets unchanged.

## Endpoint-fitting compiler milestone

The original roundAndExpand, evalPermutation4 and evalPermutation3 bodies are retained in dxt-evaluate-kernels.cuh, with their original tables and colour metric. Supporting compiler changes preserve typed local scalar pointers forwarded through device helper chains, allow flat numeric constant-vector initializers with zero-filled omitted components, and implement rintf ties-to-even rounding with signed zero.

Local pointer forwarding remains restricted to same-type pointer parameters on known helpers. Pointer arithmetic, pointer-variable escapes, type mismatches, const writes and aliased writable references remain rejected. Initializers do not enable initialized struct/vector arrays. rintf does not inherit WGSL round's implementation-dependent tie choice: the generated helper explicitly chooses even integers and retains the input sign when the result is zero.

Native/WebGPU checks cover all 1,024 original permutations for each of four original image blocks at (21,98), (64,64), (100,100) and (0,0). All 16,384 endpoint components and 16,384 error-vector components match native exactly, including the solid background block. See dxt-evaluate-check.json. A separate rintf test matches all 6,152 native output bit patterns, including positive/negative half ties, negative zero and large finite floats; see round-even-check.json.

## Remaining full-compressor work

The full kernel uses a 64-thread block and a 16-thread cooperative group in loadColorBlock, colorSums, bestFitLine and sortColors. Only the first tile enters the colour-loading branch. Mapping those tile barriers directly to workgroup barriers inside that divergent branch would be invalid; the compiler needs correct tile synchronization semantics. Shared vector compound updates, helper-local shared storage and the original error-minimum selection also need complete validation. Preserve original device bodies and the original full image while resolving these requirements. Do not claim full DXT support from endpoint tests or shrink the kernel to 16 threads.

Once the complete compression path is validated, add a GPU-result DXT1 image preview and its own main-page card linking to the sandbox. The native output reference is now available for that comparison.
