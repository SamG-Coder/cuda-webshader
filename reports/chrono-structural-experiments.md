# Chrono structural performance investigation

Measured on 13 September 2026 against `6bcf24c`, using real NVIDIA WebGPU on the RTX 5080. The competing simulation tab was closed. These are exploratory single runs, not statistically established speedups.

| Configuration | 3,000 steps |
| --- | ---: |
| Current implementation | 12.545 s |
| GPU capacity-gated neighbour fill queued before awaiting count readback | 12.519 s |
| 64-thread integration workgroups instead of 128 | 12.772 s |
| Fixed integer/boolean parameters specialized in generated WGSL | 12.540 s |

All three experiments matched the baseline's final particle values and every neighbour count exactly. Compilation is outside the timed interval; final state readback is included. The overlap experiment additionally failed the rendered sandbox with a destroyed-buffer error. It is not ready for use even apart from its negligible performance difference.

None of these production changes was retained. The restored implementation passed the rendered preview benchmark and stop/settle check at 245.3 steps/s. No native benchmark was rerun. Details are in [the measurement record](chrono-structural-experiments.json).

## Next architectural candidate: several GPU-controlled steps per submission window

This is a design hypothesis, not an implemented feature or a promised speedup. The flat results above do not establish a precise bottleneck; a current timestamp/CPU trace should accompany this experiment.

The current loop returns to JavaScript every step to inspect active count, neighbour capacity and diagnostic status. A larger change would move these scheduling decisions onto the GPU and let the host submit a small bounded window of complete steps. Every step would still run the original CUDA kernels and rebuild its own neighbours.

Required pieces:

- Capacity-sized scratch state and GPU-fed active counts throughout integration, including the empty-state path.
- A generic GPU launch gate, independent of Chrono names or physics, that blocks dependent dispatches on insufficient capacity or diagnostic failure.
- A retained first-error record and completed-step counter. Later steps must not overwrite a failure or consume incomplete neighbour data.
- A resumable capacity-growth boundary: preserve the prepared state, enlarge the buffer on the host, and resume before advancing physics. Do not silently truncate lists or replay already-completed integration.
- A bounded submission window so Stop remains responsive and rendering can share the GPU.
- Exact comparisons against the current loop, including growing buffers, shrinking active sets, empty/resumed states, injected diagnostics, and edited CUDA source reaching the GPU.

The compiler must remain generic. No replacement fluid kernels, fixture-specific equations, larger timestep, or reused neighbour lists belong in this optimization.
