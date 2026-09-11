# Architecture and invariants

## Compilation

`parser.js` tokenizes comments, identifiers, literals, operators and a restricted numeric-define preprocessor. A recursive-descent statement parser and Pratt expression parser produce an AST with source offsets. The emitter performs its own scoped symbol/type analysis and emits an ABI alongside the WGSL text.

Compilation is not NVCC/LLVM and does not translate PTX. CUDA host launches are provided separately as JavaScript workgroup counts. A single source may contain multiple kernels; the entry option selects one. Device helpers use scalar/vector values only. The UI compiler runs in a worker so source parsing does not block animation. Actual GPU pipeline creation remains asynchronous on the WebGPU device.

The emitter generates module-scope storage arrays for pointer parameters and module-scope workgroup arrays for `__shared__`. Buffer access is conservatively classified: buffers written by the kernel are writable; read-only inputs use a read-only storage binding. Scalar arguments occupy a 16-byte-rounded uniform struct with explicit offsets in metadata. Buffer ABI strides are 4 bytes for scalar f32/i32/u32, 8 for float2, and 16 for float4. Packed CUDA float3 pointers are rejected because a silent stride mismatch would corrupt addressing.

Identifiers are prefixed by role to avoid WGSL keywords and accidental cross-scope collisions. Numeric operands and casts have concrete types. Ternaries lower through temporary variables and structured branches. This preserves selected-branch behavior for guarded loads and atomics, unlike mechanically rewriting every conditional to `select`.

## Synchronization

A CUDA block maps to a WebGPU workgroup. `threadIdx`, `blockIdx`, `blockDim` and `gridDim` map to invocation/workgroup builtins or the declared workgroup-size constant. Shared-only examples use `workgroupBarrier()`. A kernel that statically reads and writes storage and calls `__syncthreads()` also emits `storageBarrier()` conservatively at that barrier.

Neither barrier synchronizes different workgroups. Reduction across blocks is decomposed into dispatch levels. Each dispatch consumes the preceding level's storage output on the same queue. No CPU access occurs between levels.

Do not put a barrier behind nonuniform control flow. All live invocations in a workgroup must meet compatible barrier control flow; the browser validator is authoritative. The CPU oracle can detect mismatched barrier sites/early exits in the fixtures but is not a complete static uniformity analysis. There is no translation of cooperative grid-wide synchronization.

The frontend also parses local `cooperative_groups::thread_block` handles
constructed with `this_thread_block()`, including explicit namespace aliases.
Handles are compile-time, scoped symbols without shader storage. `sync(handle)`
and `handle.sync()` lower to the existing barrier path, preserving storage
visibility analysis and CPU-oracle synchronization sites. Other group types,
numeric use of handles, helper-local handles and out-of-scope handles are rejected.

Direct function-forwarding macros are recorded during tokenization and resolved
at call sites. Definitions do not apply retroactively, argument arity is checked
at every forwarding stage, and chains are bounded to 32 calls. Each argument
appears once in the resulting AST. This is a restricted macro grammar, not a C++
preprocessor. Unsupported substitutions and recursive chains fail explicitly.

Signed/unsigned 24-bit multiply intrinsics emit small WGSL helper functions. The
helpers mask or sign-extend inputs and use 32-bit multiplication. Function
parameters prevent the shader compiler from rejecting overflowing literal
arguments as constant-expression overflow. The CPU oracle uses integer multiply;
independent BigInt and native 64-bit references verify boundary cases.

## Resource ownership

`GpuRuntime.createBuffer()` owns an allocation; `importBuffer()` borrows one. Owned allocations are destroyed by the runtime. Borrowed allocations are never destroyed by runtime cleanup, although their wrapper can be invalidated. Raw imported buffers must belong to the same device; WebGPU does not expose a generic buffer-to-device identity query, so the Three bridge explicitly compares renderer and runtime devices before importing.

The bridge first asks Three r186 to create a `StorageInstancedBufferAttribute`, then borrows its existing backend `GPUBuffer`. Compute writes the very same allocation that the material reads. The bridge is deliberately isolated and version-checked because `backend.get(attribute).buffer` is not a stable cross-version public interoperability contract.

Never set `attribute.needsUpdate` after GPU writes unless an intentional overwrite from the CPU seed is wanted. The array passed at initialization is stale once the simulation advances. The UI's readback counter includes tests; test readbacks do not imply that the live frame path performs readback.

Callers must not destroy/reuse resources while recorded but unsubmitted work still refers to them. Explicitly await `runtime.idle()` before replacing a live scene's buffers. Do not dispose a runtime while a renderer sharing its device is still active.

## Parameter snapshots and batching

A bind group refers to a shared uniform arena with a dynamic offset. Each recorded parameter version receives an aligned slot copied into batch-owned CPU memory. Repeated dispatches with unchanged invocation parameters reuse a slot and avoid redundant `setPipeline`/`setBindGroup` commands.

At submit, one `queue.writeBuffer()` uploads the batch's used arena range and then its command buffer is submitted. Both operations are ordered on the same queue. A subsequent batch's upload is placed after the preceding batch's commands, so the arena can be reused without waiting for a fence. Two independently recorded batches retain separate CPU snapshots even when submitted in reverse recording order.

This is specifically designed to avoid the common bug where several recorded dispatches all observe the last scalar values written to one uniform buffer. There are regression tests for repeated versions, unchanged reuse, interleaved recording, transactional update failures and arena exhaustion.

ComputeBatch has explicit submit/discard lifetime. Workgroup limits and resource lifetime are checked before dispatch. Raw buffer bounds relative to scalar dimensions cannot be inferred in general; the reusable operation plans add known shape checks for SAXPY, matrix multiplication and reduction.

## Performance scope

Optimized sample kernels are algorithms supplied in CUDA source. The translator does not automatically discover tiling or fuse arbitrary kernel graphs. It preserves the sample's organization and relies on the browser/driver compiler for machine-code generation. The separately supplied tuner evaluates real variants; it does not report hypothetical CUDA-equivalent speed.

The default matrix kernel stages 16×16 tiles in shared memory. The register variant computes four outputs per lane, trading fewer invocations/more reuse against greater register pressure. Transpose uses a padded shared tile; the benefit is architecture-dependent. The reduction loads two values per lane and uses successively smaller scratch buffers. Shared histogram bins reduce the number of global atomic updates. None of these techniques guarantees a speedup on every adapter.

Normal animation uploads a tiny parameter block, encodes one simulation dispatch, and renders instanced sprites. There is no full-particle upload after initialization, no per-frame GPUBuffer allocation, no frame-loop CPU readback, and no per-frame completion wait. CPU command encoding, sprite overdraw, browser scheduling and compositor overhead remain real costs.

## Verification layers

1. Parser/type/negative tests ensure unsupported constructs and ABI hazards are rejected in the exercised cases.
2. The CPU oracle executes the same parsed AST in typed lanes, independently of the WGSL emitter, and compares with independent mathematical reference functions.
3. Mock GPU tests verify host data packing, binding reuse, submission ordering and disposal. They do not run a shader.
4. C++ syntax checks verify the sample sources are parseable with CUDA declarations stubbed. They do not prove CUDA synchronization semantics.
5. The browser suite asks the actual WebGPU compiler to compile emitted WGSL, dispatches it and checks readback/canaries.
6. The Three integration test moves a shared position with a translated kernel, renders before/after, and tests the resulting pixels.
7. Optional NVCC tests execute selected original CUDA kernels against host references on an NVIDIA GPU.

Only layers 1–4 were available and executed in the authoring environment. Layers 5–7 must not be inferred from those successes. In particular, the AST oracle and syntax check cannot establish that every emitted shader passes a browser's WGSL validation.
