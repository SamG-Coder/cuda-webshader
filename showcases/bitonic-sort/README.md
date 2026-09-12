# NVIDIA complete bitonic sort

[Open in sandbox](../../sandbox.html?example=bitonic-sort).

Original device functions from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/2_Concepts_and_Techniques/sortingNetworks/bitonicSort.cu` and `sortingNetworks_common.cuh`. All original CUDA bodies are unchanged. The sample retains its BSD-3-Clause notice.

The default preview sorts the original 1,048,576 key/value pairs in descending order. Keys are captured from the original Windows host initialization (`srand(2001)`, `rand()%65536`); values are original element indices. The MIT initialization helper fills those indices on the GPU. The unchanged shared kernel starts the network, followed by the original global/shared merge schedule. Explicit `bufferAliases` preserve native in-place merges using one WebGPU storage binding per physical buffer, with independent CUDA pointer offsets.

The display helper shows original keys on the left and sorted keys on the right, in grayscale. The original Windows random generator produces keys up to 32767, so the display range is explicitly 0..32767. Values remain attached to their keys but are not used as colours. The helper adds no sorting or replacement key values. Intermediate buffers remain on the GPU; the final image is read for presentation.

## Verification

The unchanged native program passes every original array length from 64 to 1,048,576. The native capture harness tests all 15 lengths in both directions, always with one million elements. Full-file SHA-256 hashes compare every key and value byte, including equal-key value ordering. All 30 cases match real NVIDIA WebGPU. Additional GPU checks verify independent aliased-pointer offsets, conflicting binding rejection, and ordered writes through references to the same shared element.

To regenerate reference hashes, run `tests/sorting-networks-native.cu` after building against the pinned original sample, then `node scripts/hash-sorting-captures.mjs`. Captures remain in `.local/sorting-captures`; the small published manifest avoids storing 240 MiB of repeated sorted output. `node scripts/test-sorting-networks.mjs` performs the complete comparison. After `npm run build`, `node scripts/test-sorting-sandbox.mjs` verifies the default full sort and every display pixel.

Nested expression macros are bounded to eight levels and 65,536 AST nodes, with recursion rejected. Shared-array references capture their element indices and retain sequential alias semantics. Other unsupported pointer operations remain rejected. The native odd-even merge implementation is a separate next candidate; this showcase implements the complete bitonic path.
