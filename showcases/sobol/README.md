# NVIDIA Sobol projections

[Open in sandbox](../../sandbox.html?example=sobol).

Original `sobolGPU_kernel` from NVIDIA cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/SobolQRNG/sobol_gpu.cu`. The complete CUDA body, shared direction cache, pointer offsets, Gray-code initialization and strided recurrence remain unchanged. Existing compiler support already handles this kernel.

The browser computes the original default workload: 100,000 vectors in 100 dimensions (10 million floats). Its launch matches the original host wrapper on the tested RTX 5080: 512 × 100 blocks, 64 threads per block. The direction input comes from the unchanged original CPU initializer. No native output is used as input.

The separately labelled MIT `packSobolPoints` helper copies three chosen coordinate planes into shared GPU float4 positions. The default projection uses dimensions 0, 1 and 2. In Launch settings, edit `xDimension`, `yDimension`, and `zDimension` in the second pipeline step to any index from 0 through 99, then compile and run. The full 100-dimensional output remains in the `values` buffer. Drag to orbit and scroll to zoom; this is a static sequence, not a particle simulation.

## Verification

The unchanged native program reports CPU-reference L1 error zero. Real NVIDIA WebGPU matches all 10 million native output words. Additional cases cover 1,025 vectors in 3 dimensions with 32 × 3 blocks, and 2,000 vectors in 512 dimensions with 1 × 512 blocks. These exercise tail handling and both native host launch strategies. All three cases match bit for bit.

The sandbox test compares the complete 100-dimensional output, verifies every displayed position for projections 0/1/2 and 99/37/1, checks unchanged source text and both generated shader passes. The pipeline buffer limit now permits scalar buffers up to 64 MiB, retaining the 128 MiB total allocation limit. The original 40 MB output therefore fits without reducing the workload.

The original BSD-3-Clause source includes the credited contributions from Mike Giles, Frances Y. Kuo and Stephen Joe. All notices are retained. The projection helper, host declarations and tests are MIT project code.
