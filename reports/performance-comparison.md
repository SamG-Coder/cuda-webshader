# Measured CUDA versus WebGPU — RTX 5080

Measured 2026-09-11T21:32:54.559Z. Windows WDDM, NVIDIA driver 616.64, CUDA Toolkit 13.3, NVCC -O3 -std=c++17 -arch=native, Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/152.0.0.0 Safari/537.36 Edg/152.0.0.0. WebGPU reports nvidia / blackwell; nvidia-smi and native CUDA identify the RTX 5080, the only NVIDIA GPU in this machine.

All 20 workloads passed independent numerical references on both APIs. The same ten original CUDA kernel files run through NVCC and through this project's CUDA-to-WGSL compiler. Float32 tolerance is 0.001 absolute + 0.001 relative; integer histograms must match exactly. The separate 79-case WebGPU suite checks edge cases, and five additional native CUDA checks cover tails and hierarchical reduction.

## Timing contract

Values below are median microseconds per complete operation; lower is faster. Nine samples follow five warm-up operations. Each sample batches 2,048 operations (512 for matrix multiplication). WebGPU timestamps surround the compute pass; CUDA events surround a pre-instantiated, uploaded graph containing the matching operations. CUDA graph creation and WebGPU shader/pipeline compilation are excluded. Both APIs reuse GPU-resident buffers, reset inputs before each sample outside timing, and execute the same sequence within each batch. SAXPY accumulates y and particles advance with fixed dt/time during the batch. Reduction includes every level; histogram includes a 256-thread clear kernel before each histogram. Particle measurements cover compute only.

The GPU ratio is WebGPU / CUDA graph time: above 1 means CUDA was faster. These are batched execution costs, not isolated instruction latency. CUDA ordinary launches are measured separately and include gaps caused by CPU submission. WebGPU wall time includes JS command encoding, uniform upload, submission and completion waiting. CUDA graph wall time excludes graph construction; therefore those wall columns describe different host workflows, not equal CPU overhead. Allocation, compilation, initial uploads and result downloads are excluded from all timing columns.

Repeated buffers can reside in cache; logical bandwidth is not DRAM bandwidth. Clock changes, WDDM scheduling and browser timestamp quantization remain sources of noise. Raw sample ranges are available in the HTML and JSON. Near-equal results are not evidence of a universal winner. No cuBLAS, tensor cores, transfer-heavy workload or rendering FPS comparison was performed.

## Small workloads

262,144 vector elements; 128×128 matrices; 512×512 transpose; 65,536 particles.

| Kernel | WebGPU GPU µs | CUDA graph GPU µs | WebGPU / CUDA | CUDA ordinary GPU µs | WebGPU wall µs | CUDA graph wall µs |
|---|---:|---:|---:|---:|---:|---:|
| saxpy | 1.63 | 1.49 | 1.09× | 6.66 | 3.31 | 1.50 |
| saxpy_vec4 | 1.44 | 1.40 | 1.03× | 6.46 | 2.96 | 1.40 |
| matmul_naive | 19.97 | 6.62 | 3.01× | 10.25 | 24.87 | 6.64 |
| matmul_tiled | 3.20 | 2.88 | 1.11× | 7.00 | 5.52 | 2.90 |
| matmul_register | 5.89 | 3.06 | 1.92× | 8.00 | 11.11 | 3.07 |
| reduce_sum | 3.68 | 3.55 | 1.04× | 18.63 | 7.78 | 3.55 |
| convolution | 2.05 | 1.81 | 1.13× | 6.34 | 3.10 | 1.82 |
| histogram | 4.10 | 3.82 | 1.07× | 13.90 | 7.10 | 3.82 |
| transpose | 1.76 | 1.21 | 1.46× | 6.73 | 3.07 | 1.21 |
| particles | 1.44 | 1.74 | 0.83× | 6.74 | 3.31 | 1.74 |

## Large workloads

4,194,304 vector elements; 512×512 matrices; 2048×2048 transpose; 262,144 particles.

| Kernel | WebGPU GPU µs | CUDA graph GPU µs | WebGPU / CUDA | CUDA ordinary GPU µs | WebGPU wall µs | CUDA graph wall µs |
|---|---:|---:|---:|---:|---:|---:|
| saxpy | 16.26 | 16.16 | 1.01× | 18.88 | 17.51 | 16.19 |
| saxpy_vec4 | 10.02 | 10.07 | 0.99× | 12.67 | 11.25 | 10.08 |
| matmul_naive | 231.42 | 107.18 | 2.16× | 109.64 | 235.83 | 107.20 |
| matmul_tiled | 64.26 | 63.08 | 1.02× | 65.20 | 66.86 | 63.11 |
| matmul_register | 46.21 | 43.08 | 1.07× | 45.82 | 49.00 | 43.19 |
| reduce_sum | 20.61 | 18.71 | 1.10× | 25.36 | 25.57 | 18.71 |
| convolution | 21.09 | 16.39 | 1.29× | 18.99 | 22.23 | 16.40 |
| histogram | 24.70 | 19.86 | 1.24× | 23.36 | 27.85 | 19.87 |
| transpose | 13.41 | 9.16 | 1.46× | 11.16 | 15.04 | 9.17 |
| particles | 4.51 | 4.93 | 0.91× | 7.68 | 6.22 | 4.94 |

## Reproduce

From the project directory in PowerShell:

```powershell
npm ci
npm test
$env:CW_CHROMIUM='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$env:CW_BENCH='1'
npm run test:gpu
npm run bench:native:build
.\reports\native-reference.exe
.\reports\native-benchmark.exe > reports/comparison-cuda.jsonl 2> reports/native-checks.log
npm run bench:webgpu
npm run bench:report
```

Run the APIs sequentially with the live particle app and other GPU-heavy applications closed. The build script accepts -VcVars for a different Visual Studio installation. Set CW_CHROMIUM to your preferred WebGPU Chromium browser. This report records this run, not prefilled estimates.

## Implementation and sources

- benchmarks/cases.js: shared deterministic data and independent CPU formulas.
- benchmarks/native-support.cuh: actual CUDA launches, verification, CUDA events, graph replay and ordinary launch timing.
- reports/comparison-inputs/native-generated.cu: concrete runnable launches for all twenty cases, generated from the shared manifest.
- benchmarks/webgpu.js: matched WGSL execution, readback checks and timestamps.
- comparison-webgpu.json and comparison-cuda.jsonl: every raw timing sample.
- [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): predefining a workflow reduces repeated CPU launch costs.
- [NVIDIA CUDA event timing](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/): event-based GPU timing and synchronization boundaries.
- [WebGPU specification](https://gpuweb.github.io/gpuweb/): timestamp queries and API execution semantics.
