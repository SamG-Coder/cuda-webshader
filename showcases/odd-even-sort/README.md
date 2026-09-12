# NVIDIA odd-even merge sort

[Open in sandbox](../../sandbox.html?example=odd-even-sort).

Unchanged device functions from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/2_Concepts_and_Techniques/sortingNetworks/oddEvenMergeSort.cu` and the original comparator/macros in `sortingNetworks_common.cuh`. The source retains its BSD-3-Clause notice.

The default pipeline sorts the original 1,048,576 key/value pairs in descending order. Keys come from the same original Windows host initialization as the bitonic showcase (`srand(2001)`, `rand()%65536`), and the value of each input pair is its element index. The MIT initialization helper fills those indices on the GPU.

The original 512-thread shared sort handles 1024-element tiles, followed by all 155 global merge passes from the native host wrapper. Explicit buffer aliases preserve in-place updates, including elements untouched by a particular pass. No extra copies, replacement sorting algorithm or CUDA-body edits are used. The display helper presents original keys on the left and sorted keys on the right, using the original input's 0..32767 grayscale range.

## Verification

The native capture harness invokes the unchanged `oddEvenMergeSort` host wrapper for every power-of-two array length from 64 through 1,048,576, in both directions, always sorting one million pairs. It checks key order and original key/value associations. Full-file SHA-256 hashes compare every key and value byte, including equal-key value order. All 30 cases match real NVIDIA WebGPU.

The sandbox independently matches the default native key/value hashes and verifies all 2,097,152 preview pixels. Only the final image is read for display; intermediate sorting data remains on the GPU. The full bitonic path is a separate showcase and uses its own native references.

To regenerate native manifests, run `tests/odd-even-native.cu` linked with the pinned `bitonicSort.cu` for its original `factorRadix2` helper, then `node scripts/hash-odd-even-captures.mjs`. Raw native captures stay in `.local/odd-even-captures`. Run `node scripts/test-odd-even.mjs` for the complete hardware comparison; after `npm run build`, run `node scripts/test-odd-even-sandbox.mjs` for the preview test.

The host pipeline, input-index/display helpers and test tooling are MIT project code. NVIDIA's kernels retain their original BSD-3-Clause licensing.
