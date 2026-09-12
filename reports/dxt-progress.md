# DXT texture compression candidate

Selected NVIDIA cuda-samples dxtc at pinned revision 5443602d89ed99aede2e4b7bf329daddeadb320e. Target the complete original 512-square teapot compression, all 16,384 4x4 DXT1 blocks, native comparison and a standalone sandbox showcase with decoded compressed-image preview. The complete compressor and decoded-image sandbox now run on real NVIDIA WebGPU and match native output.

## Original native baseline

Compiled and ran the unchanged dxtc.cu program on NVIDIA Blackwell using nvcc -O3 --fmad=false -std=c++17 -arch=native, the original Common headers and original teapot512_std.ppm / teapot512_ref.dds inputs. It reports Test passed with RMS(reference,result)=0.007950 against the original 0.02 threshold. See dxt-native.txt and dxt-native.dds. The program writes a DDS output beside its input, so it ran against copies in .local/dxtc-run/data and left the pinned source assets unchanged.

## Endpoint-fitting compiler milestone

The original roundAndExpand, evalPermutation4 and evalPermutation3 bodies are retained in dxt-evaluate-kernels.cuh, with their original tables and colour metric. Supporting compiler changes preserve typed local scalar pointers forwarded through device helper chains, allow flat numeric constant-vector initializers with zero-filled omitted components, and implement rintf ties-to-even rounding with signed zero.

Local pointer forwarding remains restricted to same-type pointer parameters on known helpers. Pointer arithmetic, pointer-variable escapes, type mismatches, const writes and aliased writable references remain rejected. Initializers do not enable initialized struct/vector arrays. rintf does not inherit WGSL round's implementation-dependent tie choice: the generated helper explicitly chooses even integers and retains the input sign when the result is zero.

Native/WebGPU checks cover all 1,024 original permutations for each of four original image blocks at (21,98), (64,64), (100,100) and (0,0). All 16,384 endpoint components and 16,384 error-vector components match native exactly, including the solid background block. See dxt-evaluate-check.json. A separate rintf test matches all 6,152 native output bit patterns, including positive/negative half ties, negative zero and large finite floats; see round-even-check.json.

## Full compression and sandbox verification

The original 64-thread compressor now completes all 16,384 blocks of the 512-square image. `reports/dxt-compress-check.json` records zero differing words and zero differing blocks against `dxt-native.dds`, with 24,832 nonzero compressed words. The check requires an NVIDIA adapter and validates the native capture size and nonzero content before comparing. No CUDA kernel body was changed.

The colour-loading stage also matches native for all 1,048,576 colour components, 1,048,576 sum components and 262,144 ranks, including split dispatches with a nonzero block offset. Tile participants are predicated while all 64 workgroup threads execute barriers; shared compound updates capture their reads before stores. A separate compiler fix preserves scientific notation for large float literals instead of appending an invalid decimal point after the exponent.

The sandbox pipeline uses the original packed input and permutations, runs the unchanged compressor, then executes a separately labelled MIT CUDA display decoder. All 262,144 RGBA pixels match an independent decoder of the native DDS. Compressed output remains on the GPU between dispatches; no native output is used as browser input. The main showcase card links only to `sandbox.html?example=dxt`.

The bounded tile lowering validates group parents and handle uses, first-tile guards, literal loop controls, pointer arguments, and absence of early returns. It rejects arbitrary tile scheduling and group-handle escapes. Fixed one-dimensional CUDA array parameters lower to pointer parameters, preserving the language's parameter adjustment semantics. See `tests/dxt-compiler.test.mjs` and the full-image hardware checks for coverage.
