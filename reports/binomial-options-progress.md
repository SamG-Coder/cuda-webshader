# NVIDIA binomial options: compiler bring-up

Candidate: cuda-samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`, `cpp/5_Domain_Specific/binomialOptions/binomialOptions_kernel.cu`.

The target retains the original 2048 time steps and maximum 1024-option batch. The CUDA kernel body must remain unchanged. Native CUDA capture and the original CPU comparison now pass. WebGPU comparison, sandbox pipeline and showcase publication are still outstanding.

The original launch divisibility guard, `#if NUM_STEPS % THREADBLOCK_SIZE`, initially failed preprocessing. The compiler now evaluates bounded signed 32-bit integer arithmetic in conditional directives using its existing integer-expression parser. Regression tests cover NVIDIA's valid/invalid launch guard, arithmetic precedence, undefined macros, division by zero, overflow and unsupported expressions.

The next confirmed blocker is the original mutable device-global declaration:
`static __device__ real d_CallValue[MAX_OPTIONS];`
The parser currently treats `__device__` as a function qualifier and expects a parameter list. Supporting this requires explicit storage metadata and runtime bindings for mutable device globals, rather than changing the CUDA kernel to take an output pointer.

The original input is an array of constant structs, `d_OptionData[MAX_OPTIONS]`; its full-size uniform layout and runtime upload also need validation after the output declaration is supported. Original native host preprocessing should generate reference inputs, and native outputs must be compared before a showcase card is added.


## Native reference captured

The MIT harness `tests/binomial-native.cu` includes the pinned original kernel and host wrapper without editing their bodies, and links the original `binomialOptions_gold.cpp` CPU implementation. It uses the original Windows host input generation: `srand(123)`, 1024 options, and the original S/X/T ranges with R=0.06 and V=0.10.

All 1024 outputs are finite and nonnegative. Against the original CPU implementation, relative L1 error is 5.20082632e-08 and maximum absolute error is 1.14440918e-05, within NVIDIA's original 5e-4 L1 tolerance. The run retains 2048 time steps and uses real native CUDA on the NVIDIA GPU.

The capture stores the original input structs, the exact preprocessed constant structs read back from the CUDA symbol, the native results, and CPU results. `binomial-options-native-manifest.json` records sizes and SHA-256 hashes. GPU comparisons should consume these captured inputs rather than independently approximating host preprocessing.

Reproduce from an x64 Visual Studio developer shell:

```text
nvcc -O3 --fmad=false -std=c++17 -arch=native -Xcompiler /Zc:preprocessor -I.local/nvidia-audit/Common tests/binomial-native.cu .local/nvidia-audit/cpp/5_Domain_Specific/binomialOptions/binomialOptions_gold.cpp -o .local/nvidia-checks/binomial-capture.exe
.local\nvidia-checks\binomial-capture.exe
```

Remaining compiler work includes writable device-global arrays and the full 1024-entry constant struct array. This report does not claim that the WebGPU sample works yet.
