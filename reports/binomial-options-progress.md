# NVIDIA binomial options: compiler bring-up

Candidate: cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/binomialOptions/binomialOptions_kernel.cu`.

The target retains the original 2048 time steps and maximum 1024-option batch. The CUDA kernel body must remain unchanged. Native capture, hardware comparison, sandbox pipeline and showcase publication are still outstanding.

The original launch divisibility guard, `#if NUM_STEPS % THREADBLOCK_SIZE`, initially failed preprocessing. The compiler now evaluates bounded signed 32-bit integer arithmetic in conditional directives using its existing integer-expression parser. Regression tests cover NVIDIA's valid/invalid launch guard, arithmetic precedence, undefined macros, division by zero, overflow and unsupported expressions.

The next confirmed blocker is the original mutable device-global declaration:
`static __device__ real d_CallValue[MAX_OPTIONS];`
The parser currently treats `__device__` as a function qualifier and expects a parameter list. Supporting this requires explicit storage metadata and runtime bindings for mutable device globals, rather than changing the CUDA kernel to take an output pointer.

The original input is an array of constant structs, `d_OptionData[MAX_OPTIONS]`; its full-size uniform layout and runtime upload also need validation after the output declaration is supported. Original native host preprocessing should generate reference inputs, and native outputs must be compared before a showcase card is added.
