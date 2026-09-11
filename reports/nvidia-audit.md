# NVIDIA sample audit

Revision: 5443602d89ed99aede2e4b7bf329daddeadb320e

208 sample directories inventoried; 350 direct-source CUDA entry probes. 29 isolated kernels passed native CUDA and real NVIDIA WebGPU reference checks. These use 256 elements, a 32×32 grid, 64×64 transpose/multiplication matrices, 17 dot products of 1,537 elements, 256 BlackScholes options, 3,072 unsigned scan-update values, 1,024 threads updating ten CAS counters, or 257 records in built-in typed copies. The absolute error limit is 0.000003 except BlackScholes (0.0002 absolute plus 0.00002 relative against an independent normal-distribution reference). They are correctness checks, not performance measurements.

## Full native programs

- exited-zero: 142 sample directories
- execution-failed: 2 sample directories
- configure-excluded: 12 sample directories
- waived: 3 sample directories
- build-failed: 2 sample directories
- timeout: 1 sample directories
- platform-blocked: 12 sample directories
- environment-blocked: 34 sample directories

149 executable attempts are recorded (one sample has two executables). Exit zero is reported literally; it is not an independent correctness assertion. Initial generic argument failures were retried with source-supported test arguments; previous attempts remain in the JSON. Most runs have a 12-second limit; UnifiedMemoryPerf and tileMatmulAutotuner permit 120 seconds.

## Outstanding blockers

- **simpleCUDA2GL** — execution-failed: Reference-image run terminates with Windows exit 0xC000041D (fatal user callback exception). The interactive run previously timed out. CUDA/OpenGL callback or cleanup failure needs a native debugger; the precise fault has not been established.
- **simpleMPI** — configure-excluded: -- MPI not found - will not build sample 'simpleMPI'
- **simpleP2P** — waived: Upstream sample waived execution because the required multi-GPU/peer configuration is unavailable on this single RTX 5080 PC. See the exact device checks in its run output.
- **systemWideAtomics** — configure-excluded: -- Will not build sample systemWideAtomics - requires Linux OS
- **EGLStream_CUDA_CrossGPU** — configure-excluded: -- Will not build sample EGLStream_CUDA_CrossGPU - requires Linux OS
- **EGLStream_CUDA_Interop** — configure-excluded: -- Will not build sample EGLStream_CUDA_Interop - requires Linux OS
- **segmentationTreeThrust** — execution-failed: Computation completed but produced only level_0.ppm. Upstream validation tries to read level_00.ppm and level_09.ppm, which do not exist. This is a segmentation/output-level mismatch; the underlying algorithm cause is not established.
- **streamOrderedAllocationIPC** — configure-excluded: -- Will not build sample streamOrderedAllocationIPC - requires Linux OS
- **streamOrderedAllocationP2P** — waived: Upstream sample waived execution because the required multi-GPU/peer configuration is unavailable on this single RTX 5080 PC. See the exact device checks in its run output.
- **StreamPriorities** — configure-excluded: -- Will not build sample StreamPriorities - requires Linux OS
- **dmabufInterop** — configure-excluded: -- Skipping dmabufInterop: requires Linux (dma-buf is Linux-only). Found Windows.
- **localityDomains** — build-failed: Upstream source requires locality-domain API identifiers/fields absent from installed CUDA 13.3 headers. Actual NVCC compilation failed; see target log. A compatible newer toolkit/API revision is required.
- **localityDomainsDrv** — build-failed: Upstream source requires locality-domain API identifiers/fields absent from installed CUDA 13.3 headers. Actual NVCC compilation failed; see target log. A compatible newer toolkit/API revision is required.
- **conjugateGradientMultiDeviceCG** — waived: Upstream sample waived execution because the required multi-GPU/peer configuration is unavailable on this single RTX 5080 PC. See the exact device checks in its run output.
- **cudaNvSci** — configure-excluded: -- Will not build sample cudaNvSci - requires Linux or QNX
- **simpleD3D12** — timeout: Interactive Direct3D 12 application remained running until the 12-second limit. Its command-line parser has no automated frame-count/QA exit option. This records a launch attempt, not verified rendering or a numerical pass.
- **simpleVulkan** — configure-excluded: -- glfw3 not found - will not build sample 'simpleVulkan'
- **simpleVulkanMMAP** — configure-excluded: -- glfw3 not found - will not build sample 'simpleVulkanMMAP'
- **vulkanImageCUDA** — configure-excluded: -- glfw3 not found - will not build sample 'vulkanImageCUDA'
- **cuda-c-linking** — configure-excluded: -- Skipping the build of the cuda-c-linking sample.

The 12 Tegra samples require Jetson platform APIs. The 34 Python samples were not executed: the local python command reports no Python installation, and their CUDA Python/CuPy/framework dependencies are not configured.

## Browser results

All 29 supported isolated kernels passed on a real NVIDIA adapter. The initial direct-source scan rejected 120 sample directories; each exact compiler error is recorded in nvidia-audit.json. 38 have no extractable standalone entry (including library/host-only/template examples). 34 Python samples are not CUDA C input. The baseline had 18 translated entries from 16 directories; the transpose follow-up adds two verified entries, scalarProd adds one, BlackScholes adds its complete kernel file with two helpers, and matrixMul adds 16- and 32-wide integer template specializations. The scan follow-up adds its uniformUpdate stage; helper shared-memory pointers still block earlier scan stages. The atomic follow-up adds the original cas_atomic retry loop. The alignedTypes follow-up adds int, uint4 and float4 specializations of its copy template; custom alignment structs remain unsupported. Baseline errors describe the original import attempts, while current GPU checks describe the supported extracted kernels.

Common frontend blockers are templates/classes, unsupported cooperative-group operations, device printf/inline PTX, texture/surface types and preprocessing. A compiler rejection describes the current import path; it does not establish that the algorithm cannot be implemented in WGSL.

Each verified kernel has its own main-page card that opens the sandbox. Presets contain source and launch/input data only. The sandbox compiles the CUDA anew; CPU reference calculations stay in the separate correctness harness.

See nvidia-audit.json for all 208 entries, nvidia-native.json for per-program output and previous attempts, nvidia-build-targets.json for build outcomes, nvidia-gpu.json for numeric checks, and nvidia-isolated-native.txt for the native kernel checks. Reproduction scripts and licensing are documented in ../showcases/nvidia/README.md.
