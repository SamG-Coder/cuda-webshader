# Experimental post-emission WGSL trimming

This is an opt-in test, not a production compiler or sandbox change. CUDA source, launch scheduling, numerical routines, workgroup sizes, binding layouts, diagnostics and timesteps are unchanged. There are no Chrono names, parameter values or fluid equations in the trim implementation.

## Finding

Generated WGSL repeatedly reconstructs a large constant-parameter struct to select one field. The force-derivative shader alone is 1,203,454 ASCII characters. A conservative post-emission pass can replace proven-safe constructor projections with the selected expression and remove helper functions unreachable from entry points.

For example, given matching types:

```wgsl
// Before: construct both fields, then select one.
Pair(uniforms.x, uniforms.y).a

// After: retain the selected typed uniform value.
(uniforms.x)
```

This is runtime-value-preserving simplification, not specialization to current parameter values. Changing uniforms does not require retrimming or recompilation.

## What the pass proves, and what it leaves alone

The implementation tokenizes the emitted text, pairs delimiters, reads struct layouts, and recognizes a deliberately narrow expression subset. It is not a full WGSL parser. It should only receive valid compiler-generated artifacts; GPU validation remains required.

- Every constructor argument must be a recognized uniform field, typed literal, nested constructor or supported boolean comparison. Unknown expressions abort that rewrite.
- The selected expression must already have the field's exact declared type. Abstract numeric literals and mismatched types are not projected, avoiding changes to implicit materialization/conversion.
- Storage accesses, indexing, unknown helper calls, atomics, texture operations and mutable local reads are not discarded. Potential uniform-name shadowing disables projection through that name.
- A conservative function-reference traversal retains all shader entry points and module-level references. Only unreachable functions are removed. Bindings, struct declarations and ABI metadata remain intact.
- Arithmetic is not reassociated; floating-point identities are not applied. Division corrections, double emulation, FMA behavior, signed zero and diagnostic operations in reachable code remain.

The semantic basis is structure member selection and function argument evaluation in the [WGSL specification](https://www.w3.org/TR/WGSL/#structure-access-expr). Removing unselected constructor arguments is only justified after excluding observable effects; a shorter textual expression alone is not sufficient.

## Measurements

Real NVIDIA RTX 5080, headless Microsoft Edge, 3,000 full Chrono steps per run. Runs were sequential in ABBA order. Compilation and final particle inspection were outside simulation timing; final GPU completion was inside. The actual timing and per-shader records are in `wgsl-trim-benchmark.json`.

| Metric | Original | Trimmed |
| --- | ---: | ---: |
| Total generated source across 15 shaders | 4,013,239 characters | 766,666 characters |
| RHS shader source | 1,203,454 characters | 476,139 characters |
| Run 1 / first run of each variant | 12.478 s | 12.614 s |
| Run 2 / second run of each variant | 12.495 s | 12.462 s |
| Mean simulation time | 12.487 s | 12.538 s |

The pass performed 501 projections and removed 439 unreachable helper definitions across those shaders. Total source shrank **80.9%**. Every run matched the original run's final position/velocity/density buffer bits and every neighbour count exactly.

There is **no demonstrated simulation speedup**. The small mean difference is within the observed variation. A reasonable hypothesis is that the browser/backend already eliminates these constructions before execution; this experiment does not inspect machine code or establish identical GPU instructions.

The pass itself took 147 ms and 123 ms across all 15 shaders. First-seen pipeline creation took 9.03 s original and 8.27 s trimmed; repeated pipeline creation dropped to 117 ms and 53 ms respectively. Those large cache effects prevent claiming a reliable cold-compilation improvement from these runs. CUDA-to-WGSL frontend time is separate and unchanged by this post-pass. Pipeline timing measures the runtime's module validation/pipeline creation call, including cache effects, not a pure compiler CPU timer.

## Reproduce

Validation completed: **8 focused structural tests and 248/248 real NVIDIA GPU regression checks**, plus a separate dynamic-uniform/signed-zero check. The first suite invocation lacked the Three.js import map; the harness was corrected and the entire suite rerun successfully. The final suite report is `wgsl-trim-suite.json`. Across runtime kernel calls intercepted by that run, 10,527 artifacts were processed, including repeated/cache-hit artifacts; this is not a count of unique shader programs.

From the repository root on a machine with the configured real NVIDIA Edge setup:

```powershell
node --test tests/experiments/wgsl-trim.test.mjs
node scripts/test-wgsl-trim.mjs 3000 .local/wgsl-trim-benchmark.json
node scripts/test-wgsl-trim-suite.mjs .local/wgsl-trim-suite.json
```

The production compiler, application build and sandbox do not import this pass. It is applied only by these explicit experimental runners. The suite runner applies it at the runtime kernel boundary; tests that compile or execute through separate worker/runtime instances are not automatically covered by that wrapper.

## Follow-on compiler analysis

The next candidate is scalar replacement of aggregate temporaries across helper calls, supported by actual use/def and alias analysis. That might expose register-lifetime or redundant-load improvements that simple constructor projection does not. It should be measured against a fresh baseline, including GPU timestamp measurements of the expensive passes.

Removing uniform fields and repacking bindings is a separate ABI change, not part of this experiment. So are dead-store removal, loop-invariant code motion and branch pruning from fixed parameters. Each requires stronger analysis of storage effects, pointer aliasing, control flow, barriers and diagnostics. Floating-point algebraic simplifications require explicit numerical contracts; they cannot be inferred from a desire for faster water.
