# Automatic shader dependency trimming

The compiler now defaults to `optimize: 'dependencies'`. It still produces one
artifact and one entry point per selected CUDA kernel; it does not merge kernels,
change dispatch scheduling, or expose a native object-file linker.

```js
import {compile, serializableArtifact} from './src/compiler/compiler.js';

// Default: validate the source, emit WGSL, then remove unused helper functions.
const ordinary = compile(source, {entry: 'tracePrimary', workgroupSize: [8, 8, 1]});

// Opt-in: also prune proven integer/bool branches before generating WGSL.
const specialized = compile(source, {
  entry: 'buildGroupBounds', workgroupSize: [64, 1, 1], optimize: 'specialize'
});
console.log(specialized.metadata.optimization);

// Diagnostic comparison: retain the original unoptimized output.
const baseline = compile(source, {entry: 'buildGroupBounds', optimize: false});
```

Pass the same option through `GpuRuntime.kernel(source, options)` or the compiler
worker. Serialized artifacts retain the report. Compiler version is `0.1.1` so
version-aware caches distinguish the new output. Application caches keyed only
by CUDA source must also include compiler revision AND optimizer options.

## What happens

1. The original compiler parses and validates the complete source. Unsupported
   functions, bad types, recursion, and invalid branches are not hidden by DCE.
2. Optional specialization walks the selected kernel's reachable call graph.
   A scalar helper parameter is constant only when every reachable call agrees
   and the parameter is never assigned or addressed.
3. A direct integer/bool record field can be invariant only in a closed by-value
   program: every construction initializes it, every assignment agrees, and no
   pointer, reference, nested record, global, array, or external kernel record
   can supply or mutate it. This handles Stratum's `newSink(1, ...)` bounds mode
   without hard-coding the names `Sink`, `mode`, `newSink`, or `cityGroup`.
4. Constant `if` and conditional-expression branches are pruned; reachability is
   recomputed. At most six passes are used. Mixed callers remain generic; no
   function cloning or variant explosion is introduced.
5. The candidate is emitted again and its entire runtime metadata is compared
   with the validated baseline. Interface changes or rejected optimization cause
   an explicit reported fallback to dependency-only output.
6. The final WGSL dependency graph removes only whole unreachable functions.
   All stage entries and identifiers outside functions are roots. Bindings,
   structs, constants, overrides and retained function bodies are preserved.

The baseline validation is an intentional cost: opt-in specialization does two
JavaScript compiler passes, not two GPU pipeline compilations. Floating-point
arithmetic is not folded, integer overflow is not guessed, and expressions with
side effects are not treated as constants. Unrecognized AST forms (including
inline PTX and lowered pointer/collective machinery) make specialization fall
back conservatively; final dependency trimming still works. The WGSL scanner is
for compiler-owned WGSL, not a general WGSL optimizer or validator.

## Report

`metadata.optimization` contains the original and final UTF-8 byte sizes, removed
helper names, retained roots, function counts, and the largest transitive helper
dependencies. The specialization report includes pruned branch counts, proven
scalar arguments/record fields, and any skip or fallback reason. Dependency
weights overlap: do not sum them as independent source sizes or compilation time.

A source-byte reduction is **not** a measured pipeline-time improvement. Browser
scheduling, shader validation, backend lowering, driver optimization and caches
still matter. A pipeline promise's elapsed time can include queueing. The report
does not mislabel that as dedicated compiler CPU time.

## Stratum snapshot

Pinned source: `SamG-Coder/stratum-city` revision
`458860cb0470fa2c51516c37c1aaf34935b4e280`. Identical workgroup sizes and CUDA source
were compiled in all three modes. Sizes below are UTF-8 bytes, not estimated LOC.

| Entry | Unoptimized | Dependencies | Specialize + dependencies |
|---|---:|---:|---:|
| initCamera | 15,826 | 2,596 | 2,596 |
| stepCamera | 20,535 | 8,514 | 8,514 |
| clearQueue | 15,173 | 1,943 | 1,943 |
| prepareLots | 27,148 | 15,732 | 15,732 |
| planBounds | 16,035 | 2,805 | 2,805 |
| buildGroupBounds | 197,073 | 187,291 | 164,737 |
| reduceGroupBounds | 194,033 | 3,204 | 3,204 |
| tracePrimary | 217,872 | 215,172 | 214,201 |
| reflectPixels | 265,839 | 255,621 | 250,454 |
| shadePixels | 72,934 | 66,497 | 66,497 |
| resolveFrame | 55,303 | 19,687 | 19,687 |

`reduceGroupBounds` never needed the procedural grammar. In `buildGroupBounds`,
the fixed bounds mode removes the feature-intersection helper dependency. Primary
and reflected rays really use most of the city grammar, so their reductions are
small. This does **not** make those entire dependency trees removable, and does
not prove the remaining slow native pipeline compilation is fixed.

This repository change does not update Stratum's vendored compiler. Refresh its
compiler directory (including both new modules), rebuild/version its artifact
cache, and enable `optimize: 'specialize'` in its compile options to test that path.
The current 11 independent progress entries can remain unchanged.

## Reproduce

```sh
node --test tests/dependency-optimization.test.mjs
npm test
node scripts/report-dependency-optimization.mjs /path/to/stratum-city \
  reports/dependency-optimization-stratum.json reports/optimizer-wgsl
# Uses installed Playwright Chromium. Explicit software adapter for portable CI:
CW_SOFTWARE_GPU=1 node scripts/test-dependency-optimization-gpu.mjs --stratum
# On your Windows/RTX machine use the same script without CW_SOFTWARE_GPU.
```

The GPU regression executes the same fixtures in all three modes and compares
every output word to the unoptimized AST oracle. The separate Stratum check
validates all 33 WGSL modules; module validation is not a rendered-city equivalence
check or proof of hardware-specific pipeline speed. CI publishes the adapter and
browser, exact-output report, size report and WGSL validation results.
