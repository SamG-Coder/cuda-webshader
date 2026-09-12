# NVIDIA recursive quadtree

[Run in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=quadtree).

The original `cdpQuadtree` device classes and recursive kernel come from NVIDIA
cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`:
[upstream source](https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu).
Their bodies and BSD-3-Clause notice are retained. The compiler, scheduler,
host-equivalent setup helper and preview are MIT project code.

The sandbox uses the original 1,024 input points, 128 threads per block, maximum
depth 8 and minimum 16 points per leaf. Input files contain only the original
coordinates, zeroed node storage with the root initialized, and four compiler
buffer-reference descriptors. Native output points and nodes are test references;
the sandbox does not load them to produce the visualization.

The original kernel partitions the points, sets child bounds/ranges and queues
its recursive launches. The runtime copies each completed launch generation to
a separate immutable input queue, clears the output queue, and schedules bounded
indirect GPU dispatches. No launch counts are read by the CPU to decide which
children to execute. Overflow and pending work after the configured generation
limit are errors. Diagnostics are read after all scheduled generations finish.

The native comparison covers all 193 nodes (1,351 fields) and all 1,024 final
points, with exact equality. Active generations contain 1, 4, 16 and 27 child
launch records. The sandbox reads 72 diagnostic bytes and no geometry for its
preview. The generic renderer reads scalar x/y arrays and box records directly
from GPU storage. Colours distinguish box size in the preview; the CUDA kernel
does not output colours. Scroll to zoom, drag to pan and double-click to reset.

Validation:

```powershell
$env:CW_SOFTWARE_GPU='0'
$env:CW_CHROMIUM='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
npm test
node scripts/test-gpu.mjs
npm run build
node scripts/test-quadtree-sandbox.mjs
```

See `reports/quadtree-cdp-native.json`, `reports/quadtree-sandbox-check.json`, and
the complete recursive case in `reports/nvidia-regression-gpu.json`.
