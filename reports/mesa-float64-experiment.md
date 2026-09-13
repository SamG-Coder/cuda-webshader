# Mesa / SoftFloat multiplication experiment

This is an opt-in compiler-output experiment. Production compiler, runtime, CUDA input, physics parameters, and sandbox are unchanged.

## Source review

Reviewed `src/compiler/glsl/float64.glsl` from Mesa mirror revision `7cda7850edd103ace21aac37d416d2fdf7a282e1`:
https://github.com/chaotic-cx/mesa-mirror/blob/7cda7850edd103ace21aac37d416d2fdf7a282e1/src/compiler/glsl/float64.glsl

Downloaded file SHA256: `B8B036B63A2011FB4AC053F75F0D8EB5BD26BF235E80E603DBCC62A4D558619A`.

Mesa's `__mul64To128` uses fixed wide integer products (`umulExtended`) and carry propagation. Our existing helper instead uses 53 shift/add iterations and another 49 or 50 sticky-shift iterations. The experimental WGSL uses a fixed 4-by-4 schoolbook product with 16-bit limbs, then directly extracts the rounded significand with a sticky bit. This is an adaptation of the wide-product approach, not a wholesale Mesa port or native f64 support.

Each partial-product accumulation is bounded by `65535^2 + 65535 + 65535 = 2^32 - 1`, so it fits in a WGSL u32. The normalized inputs have 53-bit significands, giving a 105- or 106-bit product. Shifting by 49 or 50 leaves the existing packer's 53 significand bits plus three guard/round/sticky bits. All discarded bits contribute to sticky. Existing exceptional-value handling, power-of-two shortcuts, exponent handling and round-to-nearest-even packing remain intact.

The transform matches only the current compiler-owned multiplication body; an unknown modified body is refused. It is not a general shader parser and is only invoked by the explicit experiment flag.

Other findings: Mesa uses leading-zero counts for normalization and estimate/correction algorithms for square root. These remain candidates, not implemented changes. The reviewed file is not a direct replacement for our library: it uses GLSL extensions and has different NaN propagation choices. In particular, its `__fmad64` is multiply followed by add; it must not be mistaken for fused CUDA fma semantics. No `__fdiv64` implementation was found in this file.

## Licence

The reviewed file explicitly carries Berkeley SoftFloat's BSD-3-Clause notice. The experiment retains that complete notice and the pinned source reference. The project's MIT licence is unchanged; the attributed experimental module carries the retained third-party terms. It is not included in the production application.

## Validation

- Real NVIDIA WebGPU; no software adapter requested.
- 74,353 binary64 operand pairs: 8,817 existing captured inputs plus 65,536 deterministic random bit-pattern pairs.
- All GPU result bits match the original helper, including its NaN policy.
- 74,223 non-NaN products additionally match an independent JavaScript binary64 reference (including zeros, infinities and subnormals).
- 20 experimental structural tests pass, including refusal of a modified helper body and metadata preservation.
- Full Chrono runs compare final particle bits and neighbour counts at every step. No source or timestep change.
- Tests are evidence over sampled inputs, not an exhaustive proof of every binary64 pair.

## Reproduce

```powershell
node --test tests/experiments/*.test.mjs
node scripts/test-wgsl-limb-multiply.mjs
node scripts/test-wgsl-trim.mjs 3000 reports/wgsl-limb-multiply-benchmark.json --limbs
node scripts/test-wgsl-trim.mjs 3000 reports/wgsl-limb-multiply-confirmation.json --limbs --reverse
```

Timing is full loop plus final GPU completion, excluding compilation and final output inspection. Both sides use the same WGSL trimming baseline. The reverse-order run checks sensitivity to temporal drift. This is not a native CUDA comparison or a measurement of the isolated multiplication helper.

A preliminary reverse-order run was discarded because a short GPU correctness rerun overlapped it. The published confirmation was rerun separately without that overlap.

## Full simulation results

| Order | Baseline mean (ms) | Limb mean (ms) | Elapsed reduction |
|---|---:|---:|---:|
| ABBA | 12633.905 | 12023.962 | 4.83% |
| BAAB | 12580.145 | 12266.815 | 2.49% |

Each run advances 3,000 fixed steps on 30,327 markers. All eight published runs preserve final particle bits and each step's neighbour count. This is a modest repeated improvement on this machine, not proof of a universal gain or real-time simulation. The change remains test-only.
