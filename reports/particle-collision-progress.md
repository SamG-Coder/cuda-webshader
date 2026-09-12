# NVIDIA particles: collision pipeline progress

Pinned source: `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/particles`.

The original `calcHashD`, `reorderDataAndFindCellStartD`, `collideD`, grid helpers
and sphere/cell collision helpers now run through the compiler. Their device
function bodies remain unchanged in `tests/particle-collision-kernel.cuh`.
Host/library includes and the integration functor are outside this isolated fixture.

Compiler support added:
- Initialized local `volatile` value snapshots. Subsequent mutation, volatile
  pointers, shared memory and arrays remain rejected; this is not general
  volatile memory or synchronization support.
- `length` on float vectors, including the CPU oracle.

The runtime adds stable unsigned key/value sorting through compiled CUDA
bitonic passes, with no intermediate readback. It preserves input ordering for
equal keys and handles maximum uint keys, non-power-of-two counts and trailing
guards. GPU reference tests cover 1, 2, 127, 128, 129, 257 and 16,384 records.
This is a library replacement, not execution of Thrust in the browser.

Native tests use the original kernels, actual `thrust::sort_by_key`, 128- and
257-particle fixtures, and NVCC `-O3 --fmad=false -std=c++17 -arch=native`.
The fixtures deliberately contain nearby particles that collide, nonzero
velocities and a configured collider. They use 64³ cells and include a partial
final workgroup. Hashes, indices, complete cell ranges and reordered positions
and velocities match native exactly. Collision velocities differ by at most
4.656613e-10 (enforced absolute tolerance 1e-7). All pipeline stages exchange
GPU buffers; readbacks only validate their results.

Remaining work before a standalone animated showcase:
1. Support the original Thrust integration functor and its zipped position/velocity
   tuple, or an explicitly documented host/library launch adaptation that keeps
   the original integration calculation reviewable.
2. Add per-frame cell clearing and sorting to the sandbox pipeline.
3. Render the complete simulation from shared GPU position data and validate
   multiple native time steps, pause/resume, edited source and visible motion.

These results establish the collision and sorting foundation. They do not yet
claim the complete upstream particle simulation or a finished showcase.
