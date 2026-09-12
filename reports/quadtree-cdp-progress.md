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

## Checked private class members

The compiler now accepts private scalar/vector/nested-record fields and private
device methods. Access is checked against the current owning class: methods may
read private state on another instance of the same class, but kernels and
unrelated helpers cannot read fields or call private methods/constructors.
Unqualified member calls retain their receiver. Public reference indexing can
expose private array elements without granting direct field access. Protected
members remain explicitly unsupported.

The unchanged quadtree probe now passes the private-access parse gate and stops
at `Points::m_x`, a scalar pointer field. Its buffer identities need to survive
class methods, point-buffer swapping and recursive launches. This is the next
required representation change; the quadtree still has no runnable browser card.

## Captured scalar buffer references

Private class fields and constructor/method parameters can now retain mutable
float, int or uint buffer references in a persistent object arena. Each reference
stores an arena resource identity and an element offset. Buffer identities are
stable across kernels/submissions; aliases from another arena are rejected.
Reads and indexed assignments dispatch to the matching registered storage buffer.
Null/out-of-range reads yield zero and out-of-range writes are ignored; these are
bounds guards, not a claim that invalid CUDA pointer accesses have defined values.
Argument offsets and indices are evaluated once. Pointer arithmetic after capture,
const pointer captures, and vector-element captures remain unsupported.

A GPU fixture preserves the original Points class bodies and uses a small test
launch harness. Two 32-point sets retain four scalar buffers through initialization,
three update submissions, offsets and output reads. The harness copies the Points
record to a local before calling its methods: this verifies the captured-buffer
representation, not quadtree's remaining storage-reference binding requirements.
Tests also reject cross-arena pointers and check signed offsets, large unsigned
indices, and out-of-range writes. All 649 unit and 210 real GPU tests pass; Bezier
and path-tracer sandbox regressions also pass.

The complete unchanged quadtree now parses past Points and stops at the const
float2 reference returned by Bounding_box::get_max. Reference-return accessors,
remaining record bindings/constructors, and recursive launch support are still
required before this candidate gets a showcase card.

## Const field-reference getters

Const getters with exactly `return field;` retain an addressable scalar, vector
or nested-record field. Const local references bind that address once, preserving
later owner mutations instead of snapshotting a value. Access through the getter
remains const, and public getters may expose private fields without exposing
writable access. Nested mutable class methods now receive the address of their
member receiver rather than the enclosing record.

The original Bounding_box and Quadtree_node bodies pass a 32-thread GPU test:
64 nested references remain live when set_bounding_box changes the owning node.
CPU reference tests independently check the same values and rejection of writes
through const references. This verifies local owners; mutable references into
storage records still require their existing supported binding rules.

Validation: 650 unit tests, 211 real NVIDIA WebGPU tests, and the Bezier and
path-tracer sandbox regressions pass. The unchanged complete-source probe now
stops at Parameters, whose struct contains const fields and constructors. That
record representation and the remaining recursive launch machinery are still
required before adding a quadtree showcase.

## Parameters constructors and const fields

Structs with device methods/constructors now use the value-class lowering path
with public default access. Const scalar fields are initialized only through
constructor member initializers. Later direct writes and whole-record assignment
are rejected. Copy construction through a local initializer remains distinct from
assignment. Unnamed method parameters are supported, as required by NVIDIA's
Parameters copy-and-advance constructor.

The original Parameters definition passes root/child/grandchild construction on
32 real GPU threads (96 constructor calls). Results preserve the parent's depth,
toggle point selectors, multiply the level-node count and retain immutable limits.
Negative tests reject field mutation, whole-record assignment and missing const
initializers. All 651 unit and 212 real GPU tests pass, as do both existing Bezier
and path-tracer sandbox checks.

The unchanged full-source probe now stops at `volatile int *s_num_pts[4]` in the
quadtree kernel. Shared pointer-array volatility and the remaining recursive
execution interfaces still need support. No quadtree showcase is published yet.

## Volatile shared pointer arrays

The compiler accepts local arrays of volatile int/unsigned-int shared pointers,
including the original quadtree's `(volatile int *)&smem[...]` assignments.
Slots retain evaluated element offsets into one shared allocation. The allocation
uses WGSL atomic storage, and reads/writes through slots retain atomic access.
This preserves individual memory accesses; CUDA synchronization still needs its
own translated barriers. Casts cannot change element types or remove qualifiers.

A 32-lane hardware test checks 128 cross-lane values through four shared pointer
slots, with a CPU reference check and negative tests. All 653 unit and 213 real
NVIDIA GPU tests pass, plus Bezier/path-tracer sandbox regressions. The unchanged
complete source now parses through this declaration and reaches the templated
recursive launch at line 546. Full recursive execution and storage-record
references remain required; no quadtree showcase is claimed by this stage.

## Templated launches and child shared memory

The parser now recognizes a template argument followed by CUDA launch syntax.
Integer-template child launches retain a canonical specialized entry in queue
metadata and compile the child using that entry. Queued children inherit the
third launch argument as their dynamic shared allocation, with validation against
the consumer. Block and shared-memory settings can use lexically scoped const
integer expressions. Non-default streams remain unsupported.

A real GPU test launches two blocks of an integer-template child through the
scheduler, each with 128 bytes of dynamic shared memory, and checks 64 reversed
values after a child barrier. The full suite passes 656 unit tests and 214 NVIDIA
GPU tests. Bezier and path-tracer sandbox regressions pass. The original quadtree
source now reaches unsupported child record/pointer arguments. Recursive queue
execution is still pending; this is not a runnable quadtree showcase yet.

## Record launch arguments

Kernel arguments now accept scalar-only records, including nested records and
const fields. Host launch inputs use dotted scalar field names, and each GPU
invocation reconstructs its own local value. The CPU oracle similarly copies
record parameters per invocation and validates component ranges.

Queued child arguments snapshot the record expression once and store its fields
in the queue. Child kernels reconstruct their own record from that snapshot.
Pointer-bearing and array fields are not supported by this value ABI. A real GPU
test executes NVIDIA's unchanged Parameters constructor in 32 parent lanes,
launches 32 children, and checks 160 fields. Subsequent parent mutations do not
change child values. All 659 unit and 215 real NVIDIA GPU tests pass; Bezier and
path-tracer sandbox regressions also pass.

The full-source probe now explicitly binds nodes and points as class value
buffers, matching the native host allocations. It stops at the offset child-node
pointer, &children[child_offset]. Offset pointers, storage references, and full
recursive execution remain before this can be a runnable quadtree showcase.

## Offset child buffer pointers

Child launches now retain element offsets into typed parent allocations, including
local aliases and previously shifted parent pointers. Each buffer argument gets
an offset word in the queue; the child starts its pointer at the captured offset.
Negative and beyond-allocation offsets flag queue failure and do not launch work.
Const removal and unrelated/local allocations remain rejected. The same offset
initialization applies to registered scalar buffer imports.

The GPU test checks eight children writing disjoint ranges, prefix/suffix sentinel
preservation, later parent alias mutation, and rejection of two invalid offsets
without output changes. All 661 unit and 216 real NVIDIA GPU tests pass. Bezier
and path-tracer sandbox regressions pass. The Bezier queue inspection now reads
the metadata stride rather than assuming the previous record width.

The unchanged quadtree source reaches Points::get_point, which needs registered
coordinate resources in the launch harness. Full storage references and recursive
execution still remain; no quadtree showcase is published yet.

## Storage record references and coordinate setup

The probe now includes a separate setup kernel registering the four scalar
coordinate buffers and constructing the two original Points records. NVIDIA's
imported class and kernel bodies are unchanged. The compiler also resolves the
logical CUDA warpSize constant (32), while preserving local shadowing.

Mutable local references can bind writable storage records with a captured index.
Reference helper specialization retains nested record paths, so the original
Quadtree_node::set_bounding_box calls Bounding_box::set on the same stored node.
Storage writes propagate to binding access analysis. Const restrictions remain
in effect. Nested const field references retain their address after owner updates.

The GPU test updates 32 original nodes, changes the local index after binding,
checks 96 values, then checks the same values from a second dispatch. All 663
unit and 217 real NVIDIA GPU tests pass, plus Bezier/path-tracer sandbox checks.
The unchanged full source now stops at the local warp_cnts array initializer;
full recursive execution remains required before publishing a quadtree showcase.

## Local scalar array initialization

Local fixed arrays now accept brace initialization, including explicitly nested
scalar arrays. Each initializer is evaluated and stored in source order, and
omitted elements receive zero values. The compiler rejects excess elements,
narrowing conversions, unsupported element types and writes to const arrays.
The CPU oracle implements matching nested zero-fill. CUDA warpSize also works
when passed as a function argument.

A real GPU test checks 320 values across 32 lanes: incrementing initializers,
partial/nested/empty lists, const arrays and warpSize in max(). All 665 unit and
218 real NVIDIA GPU tests pass, as do the Bezier and path-tracer sandbox checks.
The original quadtree now reaches cooperative_groups::thread_block_tile<32>.
Its tiled collectives and full recursive execution remain before a runnable
quadtree showcase can be added.
