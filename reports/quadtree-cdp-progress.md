# NVIDIA cdpQuadtree progress

Upstream revision: `5443602d89ed99aede2e4b7bf329daddeadb320e`.
Source: `cpp/3_CUDA_Features/cdpQuadtree/cdpQuadtree.cu`.

The unmodified NVIDIA sample builds and runs on the RTX 5080 with CUDA 13.3.
Its own recursive validator reports `Results: OK`. Thrust/CCCL requires
`-Xcompiler /Zc:preprocessor` with this MSVC installation; the initial native build
failed without that flag. No upstream CUDA body was modified.

The native capture harness includes the original source and retains the full
1024-point, max-depth-8, minimum-16-points, 128-thread workload. Original random
point generation and original recursive kernel execution produced 193 live nodes,
145 leaves and a deepest occupied level of 4. Independent checks confirm an exact
point permutation and exactly one leaf membership for every output point.

Artifacts:
- `quadtree-native.txt`: original sample output.
- `quadtree-cdp-native.json`: capture validation.
- `quadtree-input.bin` / `quadtree-output.bin`: interleaved float2 points.
- `quadtree-nodes.json`: live nodes with bounds, ranges and depth.
- `quadtree-cdp-probes.json`: current unchanged-source compiler probe.

The importer previously extracted member functions without their class
surroundings, producing an unsupported `Points` type error. It now retains whole
referenced class/record declarations and their transitive type dependencies,
while dropping unrelated host records and their methods. Existing supported zip
functors remain runnable entries.

The next observed compiler rejection is private class fields. The source also
uses pointer-owning Points records, const reference accessors, Parameters
constructors and a by-value Parameters kernel argument. Recursive templated
child launches pass an offset node pointer and dynamic shared-memory launch
bytes. Those interfaces need investigation and implementation; the leaf-only
Bezier scheduler does not establish support for them.

No quadtree showcase card is added until the full browser workload works and
matches this native capture. This sample partitions 2D points; it is not a 3D
renderer.
