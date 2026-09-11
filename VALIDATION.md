# Validation report

Validated on 12 September 2026 (Australia/Sydney), on Windows with NVIDIA GeForce RTX 5080, driver 616.64, CUDA Toolkit 13.3 and Microsoft Edge 152.

| Check | Actual result |
|---|---|
| Node compiler, CPU oracle and host/plan tests | **144 passed; 0 failed** |
| Real browser GPU numerical suite plus Three.js shared-buffer rendered-pixel test | **79 passed; 0 failed** |
| Native CUDA tail/matrix/hierarchical reference | **5 passed** |
| Matched CUDA/WebGPU performance cases | **20/20 verified on each API**, covering all ten kernels at two sizes |
| Native CUDA compilation | All ten original kernels compiled with NVCC -O3 -std=c++17 -arch=native |
| Dependency install | Succeeded; lockfile included; npm reported zero vulnerabilities |
| Static build | Succeeded |
| Source app and static build startup | Both rendered 131,072 particles; pause works; no page errors, fatal overlay or horizontal overflow at 1440×960 |
| Comparison page | Both size filters verified; screenshot visually checked |
| Supplied CI workflow | Not executed |

The live smoke check observed 56 and 61 FPS in brief samples. These are startup frame-cadence observations, not a sustained rendering benchmark or CUDA comparison.

The previous authoring environment could not execute GPU tests or install dependencies. Those limitations do not apply to this local run. The earlier syntax-only and HTTP logs remain historical evidence; the current hardware results supersede the old NOT RUN status.

## Performance methodology and evidence

See [the complete measured report](reports/performance-comparison.md) and [the interactive table](reports/performance-comparison.html). CUDA graphs and WebGPU compute passes batch the same operations, with nine timed samples. The same deterministic input bytes and independent reference formulas are used for both APIs. CUDA event timings, ordinary CUDA launch timings, WebGPU GPU timestamps, and host wall times are retained separately. Full reduction chains and histogram clearing are included.

GPU timings exclude allocation, shader compilation, graph instantiation, initial uploads and result downloads. Particle timings measure compute only. GPU-resident reuse can benefit from caches. This is not a cuBLAS, tensor-core or end-to-end transfer benchmark. Near-equal results should be treated as near parity within this measurement, not a universal performance guarantee.

The original tuner's single-dispatch wall-time calibration selected batches too short for browser timestamp resolution. It now calibrates using a GPU-timestamped probe batch and allows longer samples. All nine tuner variants subsequently returned valid GPU timestamp medians.

- reports/node-tests.tap: 144 Node test results.
- reports/gpu-local.json: 79 GPU tests and the corrected tuner results.
- reports/native-build.log: NVCC compilation output.
- reports/native-reference.log: five native CUDA correctness checks.
- reports/native-checks.log: reference checks for twenty native workloads, including maximum errors.
- reports/comparison-manifest.json: shapes, launches, input types and sample configuration.
- reports/comparison-inputs/: shared binary inputs, expected outputs and generated real CUDA launch examples.
- reports/comparison-cuda.jsonl: all CUDA graph and ordinary launch samples.
- reports/comparison-webgpu.json: all WebGPU samples, adapter information and numerical checks.
- reports/comparison-summary.json: joined results and per-case ratios.
- reports/app-smoke.json: source/build startup and report filter checks.
- reports/live-app.png and reports/performance-comparison.png: screenshots of actual running pages.
