# Bend 2 inside CUDA WebShader

Open `bend.html` through the local server (`npm start`). The **Bend 2 lab** is
linked from the showcase explorer and CUDA sandbox. No Bend installation, Bun,
WSL, native CUDA toolkit, network compiler service, or package hub is needed to
use the browser tool.

## What the pipeline does

1. The upstream Bend 2 parser and TypeScript checker validate the source.
2. A lowering pass walks its **checked terms**, preserves closures and algebraic
   constructors, and emits a compact table describing the reachable program.
3. That table is embedded in a CUDA implementation of a functional term machine.
4. The existing CUDA WebShader compiler produces WGSL and its normal buffer ABI.
5. The existing WebGPU runtime executes `bend_run` and reads outputs and status.

This is a new runtime for Bend terms. It does not pass upstream BendRT's native
CUDA output through a text rewriter. The generated `.cu` is real CUDA, independently
compiled and executed with NVCC in the native checks. No JavaScript evaluator
calculates the displayed GPU results.

`src/bend/compiler.js` is the frontend adapter/lowering; `src/bend/runtime.cu` is
the machine; `src/bend/runtime.js` validates/encodes inputs and dispatches through
`GpuRuntime`. The UI compiles in a worker with a 15-second timeout.

## Supported execution

- Entry arguments and results: U32, F32, and Nat represented in 32 bits.
- U32 arithmetic, bitwise operations, comparisons and single/multiple-bit shifts.
  Division by zero returns zero; remainder by zero returns the dividend, matching
  the pinned Bend implementation.
- F32 add, subtract, multiply, divide, equality/less-than, negate, absolute value,
  square root, sine and cosine. Transcendental precision follows the GPU backend.
- Recursive function calls, non-tail recursion, partial application, captured
  closures, local bindings and constructor pattern matching.
- User algebraic datatypes and generic constructors such as lists, subject to
  the reachable term/primitive restrictions. Type-only arguments are erased using
  checker annotations, not guessed from source spelling.
- Laws and proofs are checked with the source. The identity-law example shows
  this; an incorrect proof prevents compilation.

The included examples demonstrate a recursively branching computation, allocating
and folding a tree, a captured F32 closure, and a checked identity law.

## Execution and resource boundaries

Each JSON input row launches one independent GPU invocation. Parallel lets inside
that computation currently evaluate sequentially. This preserves the results of
the supported pure computations; it does **not** reproduce BendRT's internal
fork/join parallel scheduling or performance. There are no performance-parity claims.

Each invocation owns a separate scratch arena. Every value, environment and
continuation is a four-word cell addressed by a 32-bit offset. Allocation is
monotonic for an invocation and reclaimed when its buffers are destroyed; there
is no per-value garbage collection. Bounds and step limits fail with a per-row
status instead of returning an apparently valid truncated result. Exhaustion in
one row does not modify another row's arena.

| Setting | Limit |
|---|---|
| Source | 100,000 characters |
| Program | 4,096 nodes / 64 KiB constant table |
| Entry arguments | 8 scalars |
| Live constructor fields | 8 |
| Input rows | 1–1,024 |
| Arena | 64–1,048,576 words per row, multiple of four |
| Total arenas | At most 128 MiB, also checked against device limit |
| Machine steps | 1–1,000,000 per row |

The host accepts finite numeric input only. U32 wraps modulo 2^32. Nat successor
past 2^32−1 explicitly fails, rather than silently adopting U32 wraparound.
Native Bend's broader Nat range is not supported. Arrays/IO cannot cross the
scalar entry ABI. Bit-constructor patterns for U32/F32 are rejected. Foreign code,
unsafe definitions, imports other than Base, missing proofs, and TODOs are rejected.
Other unsupported live terms or base operations produce compilation errors.

Successful source checking is not the Lean-backed `--verdict`. Neither source
checking nor these tests prove this runtime, the lowering, or CUDA-to-WGSL
translation correct. The UI and compilation artifacts identify this boundary.

## Command-line compilation

```sh
node scripts/compile-bend.mjs program.bend --entry main --out generated/bend
```

This writes the CUDA source, WGSL, normal WebShader artifact JSON, and a
`.bend.json` sidecar with the source entry signature and verification status.
The CUDA entry is `bend_run`; source entry parameters are packed as one 32-bit
word each into row-major `inputs` (F32 uses its IEEE bits). Use `runBend` for
validated allocation/launching rather than manually supplying unchecked sizes.

## Reproducible validation

```sh
npm run test:bend
npm run test:bend:gpu
node --experimental-transform-types scripts/test-bend-reference.mjs /path/to/bend
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test-bend-native.ps1
```

`CW_CHROMIUM` can select an installed Chrome/Edge executable for GPU tests. The
upstream reference check needs the Bend checkout at the pinned revision and Node
24; it executes Bend's independent JS compiler. The native check needs NVCC and
Visual Studio; generated native source/binaries are placed in the OS temporary
directory, not the library checkout. The native check uses `--fmad=false`.

The common fixture set covers 21 programs / 102 outputs, including unsigned
overflow, division/remainder by zero, shift boundaries, nested closures, generic
lists, constructors and recursion. Browser checks additionally cover dispatches
of 1/63/64/65/257 rows, arena isolation, step limits, Nat overflow, and the UI.
Results are in `reports/bend-reference.json`, `reports/bend-native.txt` and
`reports/bend-runtime-gpu.json`; the inspected screenshot is
`reports/bend-runtime.png`. Timings include dispatch and readback, and exclude
compilation; they are smoke-test measurements, not benchmarks.

## Upstream provenance

Frontend: [bendlang/bend](https://github.com/bendlang/bend), commit
`3378e6237ed431d17629efd36d24c96241815b7e` (CLI version 2.0.32), Apache-2.0.
The license and hashes are in `src/bend/vendor`. Regenerate with Node 24:

```sh
node scripts/vendor-bend.mjs /path/to/bend
```

The script verifies the input files against the pinned Git objects, normalizes
line endings, strips TypeScript types and removes the filesystem/package loader.
It preserves the parser, checker and base source. This tool's runtime and adapter
are maintained by CUDA WebShader and do not imply upstream endorsement.

## Next architectural step

For intra-program parallelism, add shared task records and join continuations,
publish new work through bounded queues, and process those queues across
dispatches. That would require a different allocation/lifetime model than the
current isolated arenas. The current runtime establishes a tested semantic
baseline before adding that scheduler or specializing hot paths into direct CUDA.
